
import onnx
def change_input_output_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "batch"
 
    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        dim1.dim_param = sym_batch_dim
        # or update it to be an actual value:
        # dim1.dim_value = actual_batch_dim
    
    outputs = model.graph.output
    for output in outputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = output.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        dim1.dim_param = sym_batch_dim
 
def change_input_node_name(model, input_names):
    for i,input in enumerate(model.graph.input):
        input_name = input_names[i]
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == input.name:
                    node.input[i] = input_name
        input.name = input_name
        
 
def change_output_node_name(model, output_names):
    for i,output in enumerate(model.graph.output):
        output_name = output_names[i]
        for node in model.graph.node:
            for i, name in enumerate(node.output):
                if name == output.name:
                    node.output[i] = output_name
        output.name = output_name
 
 
onnx_path = "yolo11s-pose.transd.onnx"
save_path = "yolo11s-pose.transd.dyn.onnx"
model = onnx.load(onnx_path)
change_input_output_dim(model)
# change_input_node_name(model, ["data"])
# change_output_node_name(model, ["fc1"])
 
onnx.save(model, save_path)
