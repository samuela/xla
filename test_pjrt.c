// Test program for XLA PJRT C API CPU plugin.
//
// Loads the CPU plugin via dlopen, creates a client, compiles a StableHLO
// program that computes axpy (alpha*x + y), executes it, and verifies results.
//
// Usage: ./test_pjrt <path-to-pjrt_c_api_cpu_plugin.so>

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#define CHECK(expr, msg)                                       \
  do {                                                         \
    if (!(expr)) {                                             \
      fprintf(stderr, "FAILED: %s (at %s:%d)\n", msg,         \
              __FILE__, __LINE__);                             \
      exit(1);                                                 \
    }                                                          \
  } while (0)

#define CHECK_OK(err, api, msg)                                \
  do {                                                         \
    if ((err) != NULL) {                                       \
      PJRT_Error_Message_Args eargs;                           \
      memset(&eargs, 0, sizeof(eargs));                        \
      eargs.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE; \
      eargs.error = (err);                                     \
      (api)->PJRT_Error_Message(&eargs);                       \
      fprintf(stderr, "FAILED: %s: %.*s\n", msg,              \
              (int)eargs.message_size, eargs.message);         \
      PJRT_Error_Destroy_Args dargs;                           \
      memset(&dargs, 0, sizeof(dargs));                        \
      dargs.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE; \
      dargs.error = (err);                                     \
      (api)->PJRT_Error_Destroy(&dargs);                       \
      exit(1);                                                 \
    }                                                          \
  } while (0)

// StableHLO MLIR module: axpy(alpha, x, y) = alpha * x + y
// alpha: f32 scalar, x: f32[4], y: f32[4] -> f32[4]
static const char* AXPY_MLIR =
    "module @axpy {\n"
    "  func.func @main(%alpha: tensor<f32>, %x: tensor<4xf32>, "
    "%y: tensor<4xf32>) -> tensor<4xf32> {\n"
    "    %bcast = \"stablehlo.broadcast_in_dim\"(%alpha) "
    "{broadcast_dimensions = array<i64>} : (tensor<f32>) -> tensor<4xf32>\n"
    "    %ax = stablehlo.multiply %bcast, %x : tensor<4xf32>\n"
    "    %result = stablehlo.add %ax, %y : tensor<4xf32>\n"
    "    return %result : tensor<4xf32>\n"
    "  }\n"
    "}\n";

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <path-to-pjrt_c_api_cpu_plugin.so>\n", argv[0]);
    return 1;
  }

  // Load plugin
  void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  CHECK(handle != NULL, dlerror());

  typedef const PJRT_Api* (*GetPjrtApiFn)(void);
  GetPjrtApiFn get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
  CHECK(get_api != NULL, "GetPjrtApi symbol not found");

  const PJRT_Api* api = get_api();
  CHECK(api != NULL, "GetPjrtApi returned NULL");
  printf("PJRT API version: %d.%d\n",
         api->pjrt_api_version.major_version,
         api->pjrt_api_version.minor_version);

  // Initialize plugin
  {
    PJRT_Plugin_Initialize_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    PJRT_Error* err = api->PJRT_Plugin_Initialize(&args);
    CHECK_OK(err, api, "PJRT_Plugin_Initialize");
  }

  // Create client
  PJRT_Client* client = NULL;
  {
    PJRT_Client_Create_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Error* err = api->PJRT_Client_Create(&args);
    CHECK_OK(err, api, "PJRT_Client_Create");
    client = args.client;
  }
  CHECK(client != NULL, "client is NULL");

  // Get platform name
  {
    PJRT_Client_PlatformName_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
    args.client = client;
    PJRT_Error* err = api->PJRT_Client_PlatformName(&args);
    CHECK_OK(err, api, "PJRT_Client_PlatformName");
    printf("Platform: %.*s\n", (int)args.platform_name_size,
           args.platform_name);
  }

  // Get first addressable device
  PJRT_Device* device = NULL;
  {
    PJRT_Client_AddressableDevices_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    args.client = client;
    PJRT_Error* err = api->PJRT_Client_AddressableDevices(&args);
    CHECK_OK(err, api, "PJRT_Client_AddressableDevices");
    CHECK(args.num_addressable_devices > 0, "no addressable devices");
    device = args.addressable_devices[0];
  }

  // Compile the MLIR program
  PJRT_LoadedExecutable* executable = NULL;
  {
    PJRT_Program program;
    memset(&program, 0, sizeof(program));
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.code = (char*)AXPY_MLIR;
    program.code_size = strlen(AXPY_MLIR);
    program.format = "mlir";
    program.format_size = 4;

    // Serialized CompileOptionsProto with num_replicas=1, num_partitions=1.
    // Proto wire format:
    //   field 3 (executable_build_options, length-delimited): 0x1a 0x04
    //     field 4 (num_replicas, varint 1): 0x20 0x01
    //     field 5 (num_partitions, varint 1): 0x28 0x01
    static const char compile_opts[] = "\x1a\x04\x20\x01\x28\x01";

    PJRT_Client_Compile_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    args.client = client;
    args.program = &program;
    args.compile_options = compile_opts;
    args.compile_options_size = 6;
    PJRT_Error* err = api->PJRT_Client_Compile(&args);
    CHECK_OK(err, api, "PJRT_Client_Compile");
    executable = args.executable;
  }
  CHECK(executable != NULL, "executable is NULL");
  printf("Compilation successful\n");

  // Create input buffers
  float alpha = 3.14f;
  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float y[4] = {10.5f, 20.5f, 30.5f, 40.5f};

  PJRT_Buffer* buf_alpha = NULL;
  PJRT_Buffer* buf_x = NULL;
  PJRT_Buffer* buf_y = NULL;

  // Helper: create buffer from host data
#define MAKE_BUFFER(data_ptr, elem_type, dims_arr, ndims, out_buf)            \
  do {                                                                        \
    PJRT_Client_BufferFromHostBuffer_Args bargs;                              \
    memset(&bargs, 0, sizeof(bargs));                                         \
    bargs.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;    \
    bargs.client = client;                                                    \
    bargs.data = (data_ptr);                                                  \
    bargs.type = (elem_type);                                                 \
    bargs.dims = (dims_arr);                                                  \
    bargs.num_dims = (ndims);                                                 \
    bargs.host_buffer_semantics =                                             \
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;                    \
    bargs.device = device;                                                    \
    PJRT_Error* berr = api->PJRT_Client_BufferFromHostBuffer(&bargs);        \
    CHECK_OK(berr, api, "BufferFromHostBuffer");                              \
    (out_buf) = bargs.buffer;                                                 \
    if (bargs.done_with_host_buffer) {                                        \
      PJRT_Event_Await_Args eargs;                                            \
      memset(&eargs, 0, sizeof(eargs));                                       \
      eargs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;                  \
      eargs.event = bargs.done_with_host_buffer;                              \
      PJRT_Error* eerr = api->PJRT_Event_Await(&eargs);                      \
      CHECK_OK(eerr, api, "Event_Await (h2d)");                               \
      PJRT_Event_Destroy_Args edargs;                                         \
      memset(&edargs, 0, sizeof(edargs));                                     \
      edargs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;               \
      edargs.event = bargs.done_with_host_buffer;                             \
      api->PJRT_Event_Destroy(&edargs);                                       \
    }                                                                         \
  } while (0)

  int64_t scalar_dims[1] = {0};  // unused, but ndims=0
  int64_t vec_dims[1] = {4};

  MAKE_BUFFER(&alpha, PJRT_Buffer_Type_F32, scalar_dims, 0, buf_alpha);
  MAKE_BUFFER(x, PJRT_Buffer_Type_F32, vec_dims, 1, buf_x);
  MAKE_BUFFER(y, PJRT_Buffer_Type_F32, vec_dims, 1, buf_y);

  // Execute
  PJRT_Buffer* inputs[3] = {buf_alpha, buf_x, buf_y};
  PJRT_Buffer* const* input_lists[1] = {inputs};
  PJRT_Buffer* outputs[1] = {NULL};
  PJRT_Buffer** output_lists[1] = {outputs};
  PJRT_Event* events[1] = {NULL};

  {
    PJRT_ExecuteOptions options;
    memset(&options, 0, sizeof(options));
    options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

    PJRT_LoadedExecutable_Execute_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    args.executable = executable;
    args.options = &options;
    args.argument_lists = input_lists;
    args.num_devices = 1;
    args.num_args = 3;
    args.output_lists = output_lists;
    args.device_complete_events = events;
    PJRT_Error* err = api->PJRT_LoadedExecutable_Execute(&args);
    CHECK_OK(err, api, "LoadedExecutable_Execute");
  }

  // Wait for execution
  if (events[0]) {
    PJRT_Event_Await_Args eargs;
    memset(&eargs, 0, sizeof(eargs));
    eargs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    eargs.event = events[0];
    PJRT_Error* err = api->PJRT_Event_Await(&eargs);
    CHECK_OK(err, api, "Event_Await (execute)");
    PJRT_Event_Destroy_Args edargs;
    memset(&edargs, 0, sizeof(edargs));
    edargs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    edargs.event = events[0];
    api->PJRT_Event_Destroy(&edargs);
  }

  printf("Execution successful\n");

  // Read output buffer to host
  float result[4] = {0};
  {
    // First call with dst=NULL to get size
    PJRT_Buffer_ToHostBuffer_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = outputs[0];
    args.dst = NULL;
    args.dst_size = 0;
    PJRT_Error* err = api->PJRT_Buffer_ToHostBuffer(&args);
    CHECK_OK(err, api, "Buffer_ToHostBuffer (size query)");
    CHECK(args.dst_size == sizeof(result), "unexpected output size");
    if (args.event) {
      PJRT_Event_Destroy_Args edargs;
      memset(&edargs, 0, sizeof(edargs));
      edargs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      edargs.event = args.event;
      api->PJRT_Event_Destroy(&edargs);
    }

    // Second call to actually copy
    memset(&args, 0, sizeof(args));
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = outputs[0];
    args.dst = result;
    args.dst_size = sizeof(result);
    err = api->PJRT_Buffer_ToHostBuffer(&args);
    CHECK_OK(err, api, "Buffer_ToHostBuffer (copy)");
    if (args.event) {
      PJRT_Event_Await_Args eargs;
      memset(&eargs, 0, sizeof(eargs));
      eargs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
      eargs.event = args.event;
      PJRT_Error* eerr = api->PJRT_Event_Await(&eargs);
      CHECK_OK(eerr, api, "Event_Await (d2h)");
      PJRT_Event_Destroy_Args edargs;
      memset(&edargs, 0, sizeof(edargs));
      edargs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      edargs.event = args.event;
      api->PJRT_Event_Destroy(&edargs);
    }
  }

  // Verify: alpha*x + y = 3.14*[1,2,3,4] + [10.5,20.5,30.5,40.5]
  //                      = [3.14,6.28,9.42,12.56] + [10.5,20.5,30.5,40.5]
  //                      = [13.64, 26.78, 39.92, 53.06]
  float expected[4] = {13.64f, 26.78f, 39.92f, 53.06f};
  printf("Results:\n");
  int pass = 1;
  for (int i = 0; i < 4; i++) {
    printf("  result[%d] = %.4f (expected %.4f)\n", i, result[i], expected[i]);
    if (fabsf(result[i] - expected[i]) > 0.01f) {
      pass = 0;
    }
  }

  // Cleanup
#define DESTROY_BUF(buf)                                             \
  do {                                                               \
    if (buf) {                                                       \
      PJRT_Buffer_Destroy_Args dargs;                                \
      memset(&dargs, 0, sizeof(dargs));                              \
      dargs.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;      \
      dargs.buffer = (buf);                                          \
      api->PJRT_Buffer_Destroy(&dargs);                              \
    }                                                                \
  } while (0)

  DESTROY_BUF(outputs[0]);
  DESTROY_BUF(buf_alpha);
  DESTROY_BUF(buf_x);
  DESTROY_BUF(buf_y);

  {
    PJRT_LoadedExecutable_Destroy_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.executable = executable;
    api->PJRT_LoadedExecutable_Destroy(&args);
  }

  {
    PJRT_Client_Destroy_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    args.client = client;
    api->PJRT_Client_Destroy(&args);
  }

  dlclose(handle);

  if (pass) {
    printf("PASS\n");
    return 0;
  } else {
    printf("FAIL\n");
    return 1;
  }
}
