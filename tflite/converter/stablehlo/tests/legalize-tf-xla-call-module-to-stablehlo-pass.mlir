// RUN: odml-to-stablehlo-opt %s -tf-xla-call-module-op-to-stablehlo-pass -cse -verify-diagnostics -split-input-file | FileCheck %s

// CHECK: module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1438 : i32}, tf_saved_model.semantics} {
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1438 : i32}, tf_saved_model.semantics} {
  // Checks that tf_saved_model.session_initializer remains unchanged.
  // CHECK: "tf_saved_model.session_initializer"() <{initializers = [@NoOp]}> : () -> ()
  // CHECK: func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
  // CHECK-DAG:   %cst = arith.constant dense<"0x31A2123EE9D688BEC6F45BBE324ED83E948540BE7C6AFBBE55ED93BEC3D02B3E3A469ABD31C1913E2BDAAC3EEDB8D6BE3FFAEFBD1A429B3E4D81953EFAD3A2BDBCAE3D3E2915D03EFB67B13D93BEA83E0D9FCF3EF590503E1D14D03E5300303E498E093ECAA5DDBEB803C7BE8CA0393E6A9D1E3E55D951BE093CD13E8DFCC83DD9A8D73E5B2FA9BECDBBD53EFDF78BBE4707753E212A083E1DDA8B3E1BBCB23E20FB793E76D39B3E0BD0ACBA35F64E3EC315A93E56CE583E395993BDC60EE1BE0BDFAD3E9CA8B93DF4393C3EE38B333E964E613E42DBE53E12F6243E6AE39ABD8418BA3E4F75E9BED513D53E3DF54ABE147485BE609907BE9E3DA5BEE56B88BEB78CF23E173AECBEA6E4993E0CEABB3EA96E97BE2386B63EBCD339BEC1EC8EBE22351EBE0F4EB7BE7F1EAF3EEB6AB13D28E7023DCBDC2D3D6FD471BDA584C83ECEC35ABED4A443BEADD793BE6A70E43E4E71D8BEC2B8DEBE705AF83EDA0ADA3EAE54653D3AD39A3D017010BEAE13DE3DE552893EEF2DF3BE3B44F2BEDB33AF3D8701B2BE6C8BBFBE32179C3EE95CCC3E9745763EC83E85BDD39CB33E6544D5BED36C333E28B3C63ECCBEBF3D1FCD76BEE5DBD73EBB6CABBE7AF025BEF37232BE126EE53E2AFFA5BD1954963D6550533EA01B07BE972F6B3EB811853D1024F83DDCCFBCBDA2319DBE97DBF3BEB3CBEABE4715B1BE04ACC3BEF269A53E8F69F4BE"> : tensor<8x16xf32>
  // CHECK:   %0 = "tf.VarHandleOp"() <{container = "", shared_name = "params.w"}> {allowed_devices = [], device = ""} : () -> tensor<!tf_type.resource<tensor<8x16xf32>>>
  // CHECK:   "tf.AssignVariableOp"(%0, %cst) : (tensor<!tf_type.resource<tensor<8x16xf32>>>, tensor<8x16xf32>) -> ()
  // CHECK:   return
  // CHECK: }
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
  func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
    %0 = "tf.VarHandleOp"() {allowed_devices = [], container = "", device = "", shared_name = "params.w"} : () -> tensor<!tf_type.resource<tensor<8x16xf32>>>
    %cst = arith.constant dense<"0x31A2123EE9D688BEC6F45BBE324ED83E948540BE7C6AFBBE55ED93BEC3D02B3E3A469ABD31C1913E2BDAAC3EEDB8D6BE3FFAEFBD1A429B3E4D81953EFAD3A2BDBCAE3D3E2915D03EFB67B13D93BEA83E0D9FCF3EF590503E1D14D03E5300303E498E093ECAA5DDBEB803C7BE8CA0393E6A9D1E3E55D951BE093CD13E8DFCC83DD9A8D73E5B2FA9BECDBBD53EFDF78BBE4707753E212A083E1DDA8B3E1BBCB23E20FB793E76D39B3E0BD0ACBA35F64E3EC315A93E56CE583E395993BDC60EE1BE0BDFAD3E9CA8B93DF4393C3EE38B333E964E613E42DBE53E12F6243E6AE39ABD8418BA3E4F75E9BED513D53E3DF54ABE147485BE609907BE9E3DA5BEE56B88BEB78CF23E173AECBEA6E4993E0CEABB3EA96E97BE2386B63EBCD339BEC1EC8EBE22351EBE0F4EB7BE7F1EAF3EEB6AB13D28E7023DCBDC2D3D6FD471BDA584C83ECEC35ABED4A443BEADD793BE6A70E43E4E71D8BEC2B8DEBE705AF83EDA0ADA3EAE54653D3AD39A3D017010BEAE13DE3DE552893EEF2DF3BE3B44F2BEDB33AF3D8701B2BE6C8BBFBE32179C3EE95CCC3E9745763EC83E85BDD39CB33E6544D5BED36C333E28B3C63ECCBEBF3D1FCD76BEE5DBD73EBB6CABBE7AF025BEF37232BE126EE53E2AFFA5BD1954963D6550533EA01B07BE972F6B3EB811853D1024F83DDCCFBCBDA2319DBE97DBF3BEB3CBEABE4715B1BE04ACC3BEF269A53E8F69F4BE"> : tensor<8x16xf32>
    "tf.AssignVariableOp"(%0, %cst) : (tensor<!tf_type.resource<tensor<8x16xf32>>>, tensor<8x16xf32>) -> ()
    "tf.NoOp"() {device = ""} : () -> ()
    return
  }
  // Checks that serving_default remains unchanged.
  // CHECK: func.func @serving_default(%arg0: tensor<1x4x8xf32> {tf_saved_model.index_path = ["inputs"]}) -> (tensor<*xf32> {tf_saved_model.index_path = ["outputs"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_inputs:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK:   %0 = "tf.VarHandleOp"() <{container = "", shared_name = "params.w"}> {allowed_devices = [], device = ""} : () -> tensor<!tf_type.resource<tensor<8x16xf32>>>
  // CHECK:   %1 = "tf.StatefulPartitionedCall"(%arg0, %0) <{config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", executor_type = "", f = @__inference_signature_wrapper_9520}> {_collective_manager_ids = [], _read_only_resource_inputs = [1], device = ""} : (tensor<1x4x8xf32>, tensor<!tf_type.resource<tensor<8x16xf32>>>) -> tensor<*xf32>
  // CHECK:   return %1 : tensor<*xf32>
  // CHECK: }
  func.func @serving_default(%arg0: tensor<1x4x8xf32> {tf_saved_model.index_path = ["inputs"]}) -> (tensor<*xf32> {tf_saved_model.index_path = ["outputs"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_inputs:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "params.w"}> {allowed_devices = [], device = ""} : () -> tensor<!tf_type.resource<tensor<8x16xf32>>>
    %1 = "tf.StatefulPartitionedCall"(%arg0, %0) <{config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", executor_type = "", f = @__inference_signature_wrapper_9520}> {_collective_manager_ids = [], _read_only_resource_inputs = [1], device = ""} : (tensor<1x4x8xf32>, tensor<!tf_type.resource<tensor<8x16xf32>>>) -> tensor<*xf32>
    return %1 : tensor<*xf32>
  }

  // CHECK: func.func private @__inference_9320(%arg0: tensor<1x4x8xf32> {tf._user_specified_name = "x"}, %arg1: tensor<i32> {tf._user_specified_name = "x"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<*xf32> attributes {tf._XlaMustCompile = true, tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4x8>, #tf_type.shape<>, #tf_type.shape<>], tf._noinline = true, tf._original_func_name = "__inference_932", tf.signature.is_stateful} {
  func.func private @"__inference_9320"(%arg0: tensor<1x4x8xf32> {tf._user_specified_name = "x"}, %arg1: tensor<i32> {tf._user_specified_name = "x"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<*xf32> attributes {tf._XlaMustCompile = true, tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4x8>, #tf_type.shape<>, #tf_type.shape<>], tf._noinline = true, tf._original_func_name = "__inference_932", tf.signature.is_stateful} {
    // CHECK:   %0 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<*xf32>
    %0 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<*xf32>
    "tf.NoOp"() {device = ""} : () -> ()
    // CHECK:   %1 = "tf.Identity"(%0) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK:   %2 = "tf.XlaSharding"(%1) <{_XlaSharding = "", sharding = ""}> {device = "", unspecified_dims = []} : (tensor<*xf32>) -> tensor<*xf32>
    %2 = "tf.XlaSharding"(%1) {_XlaSharding = "", device = "", sharding = "", unspecified_dims = []} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK:   %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<1x4x8xf32>) -> tensor<*xf32>
    %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<1x4x8xf32>) -> tensor<*xf32>
    // CHECK:   %4 = "tf.XlaSharding"(%3) <{_XlaSharding = "", sharding = ""}> {device = "", unspecified_dims = []} : (tensor<*xf32>) -> tensor<*xf32>
    %4 = "tf.XlaSharding"(%3) {_XlaSharding = "", device = "", sharding = "", unspecified_dims = []} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK:   %5 = "tf.Cast"(%2) <{Truncate = false}> : (tensor<*xf32>) -> tensor<8x16xf32>
    // CHECK:   %6 = "tf.Cast"(%4) <{Truncate = false}> : (tensor<*xf32>) -> tensor<1x4x8xf32>
    // CHECK:   %7 = call @XlaCallModule_main_0(%5, %6) : (tensor<8x16xf32>, tensor<1x4x8xf32>) -> tensor<1x4x16xf32>
    %5 = "tf.XlaCallModule"(%2, %4) {Sout = [#tf_type.shape<1x4x16>], device = "", dim_args_spec = [], module = "ML\EFR\03MLIRgoogle3-trunk\00\01\19\05\01\05\01\03\05\03\09\07\09\0B\0D\03\87e\13\011\07\0B\0F\0B\0B\0B\0B\13\13\0B33\0B\0B3\0B\0B\0B\0B\0B\0F\0B\13\0B\035\0B\0B\0B\0B\0B\0F\0B\13\1B\0B\1B\0B\0B\0F\13\0B\0B\0B\0B\13\0B\0F\0B/\13/\03\13\17\1B\1B\07\07\13\1B\1B\13\02V\03\1F\05\0F\1D\19\0F\05\11\05\13\05\15\05\17\17\1Ba\01\03\03\03\13\05\19\03\0B\07?\09I\0BK\03S\0DU\03\0B\07W\09Y\0B[\039\0D]\05\1B\05\1D\03\0B\1F;!_#a%;'c\05\1F\05!\05#\05%\05'\1D+\0F\05)\03\03/9\05+\0D\01\1D-\1D/\1D1\1D3\1F\11\01\17\01\03\05AE\0D\053C57\1D5\0D\053G57\1D7#\0D\03\03M\0D\03OQ\1D9\1D;\1D=\1D?\03\0511#\0F\03\031\1DA\1F\0B\11\02\00\00\00\00\00\00\00\03\05==\1F\0B\11\00\00\00\00\00\00\00\00)\05!A\07)\07\05\11!\07)\07\05\11A\07\09\1D)\03\05\09\11\05\01\03\03\05\11\05\03\01\03\05)\03\01\09\04\7F\05\01\11\01\11\07\03\01\09\03\11\01\15\05\03\07\0B\05\01\01\03\01\09\07\05-\03\05\05\03\01\05\04\01\03\05\03\11\05\17\05\03\07\0B\05\03\01\01\01\07\07)\1D\03\05\05\01\03\05\04\05\03\05\06\03\01\05\01\00\02\10C\11\0F\0B\09!##\11\1B\1D\1B\0F\BA\0271#71Q\1A\06#\1F\15\1D\15\13\11\1F\15\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00return_v1\00dot_general_v1\00call_v1\00sym_name\00arg_attrs\00function_type\00res_attrs\00sym_visibility\00jit_fun_flat_jax\00jit(fun_flat_jax)/jit(main)/linear.apply/linear/...y,yz->...z/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=_einsum in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>, <_PositionalSemantics.GLOBAL: 1>) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\00third_party/py/praxis/layers/linears.py\00lhs_batching_dimensions\00lhs_contracting_dimensions\00precision_config\00rhs_batching_dimensions\00rhs_contracting_dimensions\00jit(fun_flat_jax)/jit(main)/linear.apply/linear/...y,yz->...z/jit(_einsum)/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]\00callee\00jax.arg_info\00mhlo.sharding\00{replicated}\00_einsum\00args_flat_jax[0]\00args_flat_jax[1]\00jax.result_info\00[0]\00main\00public\00private\00", platforms = ["TPU"], version = 4 : i64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK:   %8 = "tf.PreventGradient"(%7) <{message = "The jax2tf-converted function does not support gradients. Use `with_gradient` parameter to enable gradients"}> {device = ""} : (tensor<1x4x16xf32>) -> tensor<*xf32>
    %6 = "tf.PreventGradient"(%5) {device = "", message = "The jax2tf-converted function does not support gradients. Use `with_gradient` parameter to enable gradients"} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK:   %9 = "tf.Identity"(%8) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %7 = "tf.Identity"(%6) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK:   %10 = "tf.Identity"(%9) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %8 = "tf.Identity"(%7) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK:   return %10 : tensor<*xf32>
    return %8 : tensor<*xf32>
    // CHECK: }
  }

  // Checks that __inference_function_9140 remains unchanged.
  // CHECK: func.func private @__inference_function_9140(%arg0: tensor<1x4x8xf32> {tf._user_specified_name = "args_0"}) -> (tensor<*xf32>, tensor<*xi32>) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4x8>], tf._original_func_name = "__inference_function_914", tf.signature.is_stateful} {
  // CHECK-DAG:   %cst = "tf.Const"() <{value = dense<> : tensor<0xi32>}> {device = ""} : () -> tensor<0xi32>
  // CHECK-DAG:   %cst_0 = "tf.Const"() <{value = dense<0> : tensor<i32>}> {device = ""} : () -> tensor<i32>
  // CHECK-DAG:   %cst_1 = "tf.Const"() <{value = dense<1073741824> : tensor<i32>}> {device = ""} : () -> tensor<i32>
  // CHECK:   %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<1x4x8xf32>) -> tensor<*xf32>
  // CHECK:   %1 = "tf.RandomUniformInt"(%cst, %cst_0, %cst_1) <{seed = 0 : i64, seed2 = 0 : i64}> {device = ""} : (tensor<0xi32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
  // CHECK:   %2 = "tf.Identity"(%1) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
  // CHECK:   return %0, %2 : tensor<*xf32>, tensor<*xi32>
  // CHECK: }
  func.func private @__inference_function_9140(%arg0: tensor<1x4x8xf32> {tf._user_specified_name = "args_0"}) -> (tensor<*xf32>, tensor<*xi32>) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4x8>], tf._original_func_name = "__inference_function_914", tf.signature.is_stateful} {
    %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<1x4x8xf32>) -> tensor<*xf32>
    %cst = "tf.Const"() {device = "", value = dense<1073741824> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %cst_1 = "tf.Const"() {device = "", value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
    %1 = "tf.RandomUniformInt"(%cst_1, %cst_0, %cst) {device = "", seed = 0 : i64, seed2 = 0 : i64} : (tensor<0xi32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
    %2 = "tf.Identity"(%1) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
    return %0, %2 : tensor<*xf32>, tensor<*xi32>
  }

  // Checks that __inference_inference_fn_9430 remains unchanged.
  // CHECK: func.func private @__inference_inference_fn_9430(%arg0: tensor<1x4x8xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "933"}) -> tensor<*xf32> attributes {tf._XlaMustCompile = false, tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4x8>, #tf_type.shape<>], tf._original_func_name = "__inference_inference_fn_943", tf.signature.is_stateful} {
  // CHECK:   %0:2 = "tf.StatefulPartitionedCall"(%arg0) <{config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", executor_type = "", f = @__inference_function_9140}> {_collective_manager_ids = [], _read_only_resource_inputs = [], device = ""} : (tensor<1x4x8xf32>) -> (tensor<*xf32>, tensor<*xi32>)
  // CHECK:   %1 = "tf.StatefulPartitionedCall"(%0#0, %0#1, %arg1) <{config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", executor_type = "", f = @__inference_9320}> {_XlaMustCompile = true, _collective_manager_ids = [], _read_only_resource_inputs = [2], device = ""} : (tensor<*xf32>, tensor<*xi32>, tensor<!tf_type.resource>) -> tensor<*xf32>
  // CHECK:   %2 = "tf.PartitionedCall"(%1) <{config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", executor_type = "", f = @__inference_tf_post_processing_9400}> {_collective_manager_ids = [], _read_only_resource_inputs = [], device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK:   %3 = "tf.Identity"(%2) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK:   return %3 : tensor<*xf32>
  // CHECK: }
  func.func private @__inference_inference_fn_9430(%arg0: tensor<1x4x8xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "933"}) -> tensor<*xf32> attributes {tf._XlaMustCompile = false, tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4x8>, #tf_type.shape<>], tf._original_func_name = "__inference_inference_fn_943", tf.signature.is_stateful} {
    %0:2 = "tf.StatefulPartitionedCall"(%arg0) {_collective_manager_ids = [], _read_only_resource_inputs = [], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_function_9140} : (tensor<1x4x8xf32>) -> (tensor<*xf32>, tensor<*xi32>)
    %1 = "tf.StatefulPartitionedCall"(%0#0, %0#1, %arg1) {_XlaMustCompile = true, _collective_manager_ids = [], _read_only_resource_inputs = [2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @"__inference_9320"} : (tensor<*xf32>, tensor<*xi32>, tensor<!tf_type.resource>) -> tensor<*xf32>
    "tf.NoOp"() {device = ""} : () -> ()
    %2 = "tf.PartitionedCall"(%1) {_collective_manager_ids = [], _read_only_resource_inputs = [], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_tf_post_processing_9400} : (tensor<*xf32>) -> tensor<*xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    return %3 : tensor<*xf32>
  }

  // Checks that __inference_signature_wrapper_9520 remains unchanged.
  // CHECK: func.func private @__inference_signature_wrapper_9520(%arg0: tensor<1x4x8xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "948"}) -> tensor<*xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4x8>, #tf_type.shape<>], tf._original_func_name = "__inference_signature_wrapper_952", tf.signature.is_stateful} {
  // CHECK:   %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1) <{config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", executor_type = "", f = @__inference_inference_fn_9430}> {_XlaMustCompile = false, _collective_manager_ids = [], _read_only_resource_inputs = [1], device = ""} : (tensor<1x4x8xf32>, tensor<!tf_type.resource>) -> tensor<*xf32>
  // CHECK:   %1 = "tf.Identity"(%0) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK:   return %1 : tensor<*xf32>
  // CHECK: }
  func.func private @__inference_signature_wrapper_9520(%arg0: tensor<1x4x8xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "948"}) -> tensor<*xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4x8>, #tf_type.shape<>], tf._original_func_name = "__inference_signature_wrapper_952", tf.signature.is_stateful} {
    %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {_XlaMustCompile = false, _collective_manager_ids = [], _read_only_resource_inputs = [1], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_inference_fn_9430} : (tensor<1x4x8xf32>, tensor<!tf_type.resource>) -> tensor<*xf32>
    "tf.NoOp"() {device = ""} : () -> ()
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    return %1 : tensor<*xf32>
  }

  // Checks that __inference_tf_post_processing_9400 remains unchanged.
  // CHECK: func.func private @__inference_tf_post_processing_9400(%arg0: tensor<1x4x16xf32> {tf._user_specified_name = "outputs"}) -> tensor<*xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4x16>], tf._original_func_name = "__inference_tf_post_processing_940"} {
  // CHECK:   %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<1x4x16xf32>) -> tensor<*xf32>
  // CHECK:   return %0 : tensor<*xf32>
  // CHECK: }
  func.func private @__inference_tf_post_processing_9400(%arg0: tensor<1x4x16xf32> {tf._user_specified_name = "outputs"}) -> tensor<*xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4x16>], tf._original_func_name = "__inference_tf_post_processing_940"} {
    %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<1x4x16xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }
  // Checks that the following functions are added.
  // CHECK: func.func private @XlaCallModule_main_0(%arg0: tensor<8x16xf32> {jax.arg_info = "args_flat_jax[0]", mhlo.sharding = "{replicated}"}, %arg1: tensor<1x4x8xf32> {jax.arg_info = "args_flat_jax[1]", mhlo.sharding = "{replicated}"}) -> (tensor<1x4x16xf32> {jax.result_info = "[0]"}) {
  // CHECK:   %0 = call @XlaCallModule__einsum_0(%arg1, %arg0) : (tensor<1x4x8xf32>, tensor<8x16xf32>) -> tensor<1x4x16xf32>
  // CHECK:   return %0 : tensor<1x4x16xf32>
  // CHECK: }
  // CHECK: func.func private @XlaCallModule__einsum_0(%arg0: tensor<1x4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<1x4x16xf32> {
  // CHECK:   %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0] : (tensor<1x4x8xf32>, tensor<8x16xf32>) -> tensor<1x4x16xf32>
  // CHECK:   return %0 : tensor<1x4x16xf32>
  // CHECK: }
}
// CHECK: }

// -----

// CHECK-LABEL: test_remove_stablehlo_custom_call_with_sharding
func.func @test_remove_stablehlo_custom_call_with_sharding(%arg: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "stablehlo.custom_call"(%arg) {
    call_target_name = "Sharding",
    api_version = 1 : i32,
    mhlo.sharding = "{replicated}"
    } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  // CHECK: return %arg0 : tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// CHECK-LABEL: test_main_func_with_undefined_visibility
func.func private @test_main_func_with_undefined_visibility(%arg0: tensor<5x10xf32> {tf._user_specified_name = "args_0"}, %arg1: tensor<10x5xf32> {tf._user_specified_name = "args_1"}, %arg2: tensor<5x5xf32> {tf._user_specified_name = "args_2"}) -> tensor<*xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<5x10>, #tf_type.shape<10x5>, #tf_type.shape<5x5>], tf._original_func_name = "__inference_function_7", tf.signature.is_stateful} {
  %0 = "tf.XlaCallModule"(%arg2, %arg1, %arg0) {Sout = [#tf_type.shape<5x5>], device = "", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "ML\EFR\03MLIR17.0.0git\00\01\19\05\01\05\01\03\05\03\09\07\09\0B\0D\03W7\0D\01+\07\0B\0F\0B3\0B\0B\0B\0B\0B3\0B\0B\0B\0B\13\0B\0F\0B\0F\0B\03\0D\0B\0B\0B\0B\0B\13\01\03\0B\03\0B\17\07\17\17\1F\02\D3\1F\01\01\11\01\00\05\0F\03\0B\0B\03\0D\03\0F\05\11\05\07\13\05\11\05\13\05\15\05\17\05\19\03\0B\17+\19/\1B+\071\1D3\05\1B\05\1D\05\1F\05!\03\03!5\05#\1D%\01\05%\1D)\01\05'\03\01\17\01#\0B\1D)\1D+\03\05--\01\09)\05\15\15\05\09)\05)\15\05)\05\15)\05\11\07\03\07\09\03\03\04]\05\01\11\01\09\07\03\01\05\03\11\01\15\05\03\0B\0F\07\03\01\07\01\09\01\05\07#\1F\03\03\05\05\03\07\06'\03\03\05\07\01\09\04\01\03\09\06\03\01\05\01\00\8E\04-\03\0B\0F\0F#\1F\15\1D\15\17A!A=\13\15\0F\0F\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00dot_v1\00add_v1\00return_v1\00sym_name\00mhlo.cross_program_prefetches\00mhlo.dynamic_parameter_bindings\00mhlo.is_dynamic\00mhlo.use_auto_spmd_partitioning\00IrToHlo.13\00arg_attrs\00function_type\00res_attrs\00sym_visibility\00precision_config\00dot.10\00add.11\00main\00\00", platforms = ["CPU"], version = 5 : i64} : (tensor<5x5xf32>, tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<*xf32>
  "tf.NoOp"() {device = ""} : () -> ()
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

  // CHECK: func.func private @XlaCallModule_main_0(%arg0: tensor<5x5xf32>, %arg1: tensor<10x5xf32>, %arg2: tensor<5x10xf32>) -> tensor<5x5xf32> {
  // CHECK:   %0 = stablehlo.dot %arg2, %arg1 : (tensor<5x10xf32>, tensor<10x5xf32>) -> tensor<5x5xf32>
  // CHECK:   %1 = stablehlo.add %0, %arg0 : tensor<5x5xf32>
  // CHECK:   return %1 : tensor<5x5xf32>
  // CHECK: }
}

// -----

// Tests that the "platform index argument" is handled from the deserialized
// StableHLO function.

// CHECK-LABEL: serialized_function_taking_platform_index_argument
// CHECK-SAME: %[[ARG_0:.+]]: tensor<1x4xf32>
func.func @serialized_function_taking_platform_index_argument(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
  // The serialized module is the following. Note the extra argument `%arg0`
  // of type `tensor<i32>`, which is the platform index argument.
  //
  //   func.func private @stablehlo_func(%arg0: tensor<i32>, %arg1: tensor<1x4xf32>) -> tensor<1x3xf32> {
  //     %0 = stablehlo.constant dense<1.000000e+00> : tensor<4x3xf32>
  //     %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [1] x [0] : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
  //     return %1 : tensor<1x3xf32>
  //   }
  %0 = "tf.XlaCallModule"(%arg0) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "ML\EFR\0DStableHLO_v0.17.3\00\01\1B\05\01\05\0B\01\03\0B\03\09\0F\13\17\1B\03M#\17\01\0F\07\0B\13\0B\0F\13\13\03\15\0B\0F\0B\0B\0B\0B\1F/\13/\01\03\0F\03\15\07\0F\17\17\17\07\13\1B\07\13\02\D1\1F\05\0F\03\03\07\09\05\11\11\01\05\17\03\13Q\17\03\13y\03\01\1F\15\01\17\01#\11\1D\13\1D\15\1F\0B\09\00\00\80?\1F\0F\11\01\00\00\00\00\00\00\00\03\05\13\13\1F\0F\11\00\00\00\00\00\00\00\00\01\02\04\09)\01\13)\05\05\11\03)\05\05\0D\03)\05\11\0D\03\1D)\03\05\0D\11\05\05\07\03\09\1B)\03\01\0D\04a\05\01Q\01\05\01\07\04O\03\01\05\03P\01\03\07\04;\03\09\0F\05\0B\0B\0F\0D\00\05B\01\05\03\0B\07F\01\07\03\09\05\03\05\09\04\01\03\07\06\03\01\05\01\00\DF\17\03\0B)\11\15\1F\19\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00constant_v1\00dot_general_v1\00return_v1\00example\00mhlo.num_partitions\00main\00\00\08%\09\05\01\01\0B\0F\15\0F\17\19\03\1B\0B\11\1D\1F\11!", platforms = ["CPU", "TPU"], version = 9 : i64}> : (tensor<1x4xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// Tests that a dummy const op is added and provided as the first operand of the call op
// CHECK: %[[DUMMY_CONST:.+]] = "tf.Const"()
// CHECK: %[[CALL:.+]] = call @[[XLA_CALL_MODULE_FUNC:.+]](%[[DUMMY_CONST]], %[[ARG_0]]) : (tensor<i32>, tensor<1x4xf32>) -> tensor<1x3xf32>
// CHECK: return %[[CALL]]

// CHECK: func.func private @XlaCallModule_main_0(%[[PLATFORM_IDX:.+]]: tensor<i32>, %[[ARG_1:.+]]: tensor<1x4xf32>) -> tensor<1x3xf32> {
// CHECK-DAG: %[[CONST:.+]] = stablehlo.constant dense<{{.*}}> : tensor<4x3xf32>
// CHECK: %[[DOT_GENERAL:.+]] = stablehlo.dot_general %[[ARG_1]], %[[CONST]], contracting_dims = [1] x [0] : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
// CHECK: return %[[DOT_GENERAL]] : tensor<1x3xf32>
