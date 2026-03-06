/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/runtime/collective_group_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/future.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/casts.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

CollectiveGroupThunk::CollectiveGroupThunk(ThunkInfo thunk_info,
                                           Thunk::Kind kind,
                                           ThunkSequence thunks, bool is_async)
    : Thunk(kind, thunk_info),
      async_execution_(is_async ? std::make_shared<AsyncExecution>(this)
                                : nullptr) {
  for (auto& thunk : thunks) {
    thunks_.emplace_back(std::move(thunk));
  }
}
absl::Status CollectiveGroupThunk::Prepare(const PrepareParams& params) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    RETURN_IF_ERROR(thunk->Prepare(params));
  }
  return absl::OkStatus();
}
absl::Status CollectiveGroupThunk::Initialize(const InitializeParams& params) {
  if (async_execution_) {
    RETURN_IF_ERROR(async_execution_->Initialize(params.execution_scoped_state,
                                                 params.executor));
  }
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status CollectiveGroupThunk::ExecuteOnStream(
    const Thunk::ExecuteParams& params) {
  int64_t async_stream_idx = Thunk::execution_stream_id().value();
  // Async streams are already assigned in gpu_executable.cc::ExecuteThunks.
  // async_streams is therefore guaranteed to be non-null and to have enough
  // elements to index by the AsyncStreamKind enum.
  se::Stream* async_stream =
      params.collective_params->async_streams.at(async_stream_idx);

  // Start() does async_stream->WaitFor(stream) + returns Guard that records
  // event on async_stream when it goes out of scope.
  ASSIGN_OR_RETURN(auto guard,
                   async_execution_->Start(params.collective_params->run_id,
                                           params.execution_scoped_state,
                                           params.stream, async_stream));

  // Gather the set of all communicators. There should be only one.
  absl::flat_hash_set<Communicator*> communicator_set;
  RETURN_IF_ERROR(Walk([&](const Thunk* thunk) -> absl::Status {
    ASSIGN_OR_RETURN(auto communicators, thunk->GetCommunicators(params));
    for (Communicator* comm : communicators) {
      communicator_set.insert(comm);
    }
    return absl::OkStatus();
  }));

  if (communicator_set.empty()) {
    return absl::InvalidArgumentError("No communicators in NCCL group");
  }
  if (communicator_set.size() > 1) {
    return absl::InvalidArgumentError(
        "More than one communicator in NCCL group");
  }

  Communicator* comm = *communicator_set.begin();
  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(comm);
  Future<> group_future = gpu_comm->GroupExecute(
      [this, &params](GpuCommunicator* comm) -> absl::Status {
        for (const std::unique_ptr<Thunk>& thunk : thunks_) {
          RETURN_IF_ERROR(thunk->ExecuteOnStream(params));
        }
        return absl::OkStatus();
      });
  RETURN_IF_ERROR(group_future.Await());

  // Guard destructor records event on async_stream.
  return absl::OkStatus();
}

absl::Status CollectiveGroupThunk::WalkNested(Walker callback) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    RETURN_IF_ERROR(thunk->Walk(callback));
  }
  return absl::OkStatus();
}

absl::Status CollectiveGroupThunk::TransformNested(Transformer callback) {
  for (std::unique_ptr<Thunk>& thunk : thunks_) {
    RETURN_IF_ERROR(thunk->TransformNested(callback));
    ASSIGN_OR_RETURN(thunk, callback(std::move(thunk)));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<CollectiveGroupThunk>>
CollectiveGroupThunk::FromProto(
    ThunkInfo thunk_info, const CollectiveGroupThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    CollectiveThunk::AsyncExecutionMap& async_execution_map,
    const Deserializer& deserializer) {
  ThunkSequence thunk_sequence;
  for (const auto& sub_thunk_proto : thunk_proto.thunks()) {
    ASSIGN_OR_RETURN(std::unique_ptr<Thunk> sub_thunk,
                     deserializer(sub_thunk_proto));
    thunk_sequence.push_back(std::move(sub_thunk));
  }

  ASSIGN_OR_RETURN(Thunk::Kind kind,
                   Thunk::KindFromProto(thunk_proto.thunk_kind()));

  bool is_async = thunk_proto.has_async_events_unique_id();
  auto thunk = std::make_unique<CollectiveGroupThunk>(
      std::move(thunk_info), kind, std::move(thunk_sequence), is_async);
  if (is_async) {
    async_execution_map[AsyncExecutionId{
        thunk_proto.async_events_unique_id()}] = thunk->async_execution();
  }
  return thunk;
}

std::optional<AsyncExecutionId> CollectiveGroupThunk::GetAsyncExecutionId()
    const {
  if (!async_execution_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to async_execution_ is unique.
  return absl::bit_cast<AsyncExecutionId>(async_execution_.get());
}

absl::StatusOr<ThunkProto> CollectiveGroupThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CollectiveGroupThunkProto* thunk_proto =
      proto.mutable_collective_group_thunk();

  std::optional<AsyncExecutionId> async_execution_id = GetAsyncExecutionId();
  if (async_execution_id.has_value()) {
    thunk_proto->set_async_events_unique_id(async_execution_id->value());
  }
  thunk_proto->set_thunk_kind(Thunk::KindToProto(kind()));

  for (const auto& thunk : thunks_) {
    ASSIGN_OR_RETURN(*thunk_proto->add_thunks(), thunk->ToProto());
  }

  return proto;
}

}  // namespace gpu
}  // namespace xla
