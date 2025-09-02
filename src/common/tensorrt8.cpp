#include "common/tensorrt8.hpp"
#include "common/check.hpp"
#include <iostream>
#include <cstring>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <stdarg.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>


namespace TensorRT8
{

using namespace std;
using namespace nvinfer1;

static class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      std::cerr << "[NVINFER LOG]: " << msg << std::endl;
    }
  }
} gLogger_;

template <typename _T>
static void destroy_nvidia_pointer(_T *ptr)
{
    if (ptr) ptr->destroy();
}

static std::string format_shape(const Dims &shape) 
{
    stringstream output;
    char buf[64];
    const char *fmts[] = {"%d", "x%d"};
    for (int i = 0; i < shape.nbDims; ++i)
    {
        snprintf(buf, sizeof(buf), fmts[i != 0], shape.d[i]);
        output << buf;
    }
  return output.str();
}

static const char *data_type_string(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT:
      return "float32";
    case nvinfer1::DataType::kHALF:
      return "float16";
    case nvinfer1::DataType::kINT8:
      return "int8";
    case nvinfer1::DataType::kINT32: 
      return "int32";
    case nvinfer1::DataType::kBOOL: 
      return "bool";
    case nvinfer1::DataType::kUINT8: 
      return "uint8";

    #if NV_TENSORRT_MAJOR >= 10
      case nvinfer1::DataType::kFP8: 
        return "fp8";
      case nvinfer1::DataType::kBF16: 
        return "bf16";
      case nvinfer1::DataType::kINT64: 
        return "int64";
      case nvinfer1::DataType::kINT4:
        return "int4";
    #endif

    default:
      return "Unknow";
  }
}

static std::vector<uint8_t> load_file(const string &file) 
{
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()) return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) 
    {
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char *)&data[0], length);
    }
    in.close();
    return data;
}

class __native_engine_context 
{
public:
    virtual ~__native_engine_context() { destroy(); }

    bool construct(const void *pdata, size_t size) 
    {
        destroy();

        if (pdata == nullptr || size == 0) return false;

        runtime_ = shared_ptr<IRuntime>(createInferRuntime(gLogger_), destroy_nvidia_pointer<IRuntime>);
        if (runtime_ == nullptr) return false;

        engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size, nullptr),
                                        destroy_nvidia_pointer<ICudaEngine>);
        if (engine_ == nullptr) return false;

        context_ = shared_ptr<IExecutionContext>(engine_->createExecutionContext(),
                                                destroy_nvidia_pointer<IExecutionContext>);
        return context_ != nullptr;
    }

private:
    void destroy() 
    {
        context_.reset();
        engine_.reset();
        runtime_.reset();
    }

public:
    shared_ptr<IExecutionContext> context_;
    shared_ptr<ICudaEngine> engine_;
    shared_ptr<IRuntime> runtime_ = nullptr;
};

class EngineImplement : public Engine 
{
public:
    shared_ptr<__native_engine_context> context_;
    unordered_map<string, int> binding_name_to_index_;

    virtual ~EngineImplement() = default;

    bool construct(const void *data, size_t size) 
    {
        context_ = make_shared<__native_engine_context>();
        if (!context_->construct(data, size)) 
        {
            return false;
        }

        setup();
        return true;
    }

    bool load(const string &file) 
    {
        auto data = load_file(file);
        if (data.empty()) 
        {
            printf("An empty file has been loaded. Please confirm your file path: %s\n", file.c_str());
            return false;
        }
        return this->construct(data.data(), data.size());
    }

    void setup() 
    {
        auto engine = this->context_->engine_;
        int nbBindings = engine->getNbBindings();

        binding_name_to_index_.clear();
        for (int i = 0; i < nbBindings; ++i) 
        {
            const char *bindingName = engine->getBindingName(i);
            binding_name_to_index_[bindingName] = i;
        }
    }

    virtual int index(const std::string &name) override 
    {
        auto iter = binding_name_to_index_.find(name);
        Assertf(iter != binding_name_to_index_.end(), "Can not found the binding name: %s",
                name.c_str());
        return iter->second;
    }

    virtual bool forward(const std::vector<void *> &bindings, void *stream,
                        void *input_consum_event) override 
    {
        return this->context_->context_->enqueueV2((void**)bindings.data(), (cudaStream_t)stream,
                                                    (cudaEvent_t *)input_consum_event);
    }

    virtual std::vector<int> run_dims(const std::string &name) override 
    {
        return run_dims(index(name));
    }

    virtual std::vector<int> run_dims(int ibinding) override
    {
        auto dim = this->context_->context_->getBindingDimensions(ibinding);
        return std::vector<int>(dim.d, dim.d + dim.nbDims);
    }

    virtual std::vector<int> static_dims(const std::string &name) override 
    {
        return static_dims(index(name));
    }

    virtual std::vector<int> static_dims(int ibinding) override 
    {
        auto dim = this->context_->engine_->getBindingDimensions(ibinding);
        return std::vector<int>(dim.d, dim.d + dim.nbDims);
    }

    virtual int num_bindings() override { return this->context_->engine_->getNbBindings(); }

    virtual bool is_input(int ibinding) override 
    {
        return this->context_->engine_->bindingIsInput(ibinding);
    }

    virtual bool set_run_dims(const std::string &name, const std::vector<int> &dims) override 
    {
        return this->set_run_dims(index(name), dims);
    }

    virtual bool set_run_dims(int ibinding, const std::vector<int> &dims) override 
    {
        Dims d;
        memcpy(d.d, dims.data(), sizeof(int) * dims.size());
        d.nbDims = dims.size();
        return this->context_->context_->setBindingDimensions(ibinding, d);
    }

    virtual int numel(const std::string &name) override { return numel(index(name)); }

    virtual int numel(int ibinding) override 
    {
        auto dim = this->context_->context_->getBindingDimensions(ibinding);
        return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<int>());
    }

    virtual DType dtype(const std::string &name) override { return dtype(index(name)); }

    virtual DType dtype(int ibinding) override 
    {
        return (DType)this->context_->engine_->getBindingDataType(ibinding);
    }

    virtual bool has_dynamic_dim() override 
    {
        // check if any input or output bindings have dynamic shapes
        // code from ChatGPT
        int numBindings = this->context_->engine_->getNbBindings();
        for (int i = 0; i < numBindings; ++i) 
        {
            nvinfer1::Dims dims = this->context_->engine_->getBindingDimensions(i);
            for (int j = 0; j < dims.nbDims; ++j) 
            {
                if (dims.d[j] == -1) return true;
            }
        }
        return false;
    }

    virtual void print() override 
    {
        printf("------------------------------------------------------\n");
        printf("Engine %p [%s]\n", this, has_dynamic_dim() ? "DynamicShape" : "StaticShape");

        int num_input = 0;
        int num_output = 0;
        auto engine = this->context_->engine_;
        for (int i = 0; i < engine->getNbBindings(); ++i) 
        {
            if (engine->bindingIsInput(i))
            num_input++;
            else
            num_output++;
        }

        printf("Inputs: %d\n", num_input);
        for (int i = 0; i < num_input; ++i) 
        {
            auto name = engine->getBindingName(i);
            auto dim = engine->getBindingDimensions(i);
            auto io_dtype = engine->getBindingDataType(i);
            printf("\t%d.%s : shape {%s} dtype {%s}\n", i, name, format_shape(dim).c_str(), data_type_string(io_dtype));
        }

        printf("Outputs: %d\n", num_output);
        for (int i = 0; i < num_output; ++i) 
        {
            auto name = engine->getBindingName(i + num_input);
            auto dim = engine->getBindingDimensions(i + num_input);
            auto io_dtype = engine->getBindingDataType(i + num_input);
            printf("\t%d.%s : shape {%s} dtype {%s}\n", i, name, format_shape(dim).c_str(), data_type_string(io_dtype));
        }
    }

};

Engine *loadraw(const std::string &file) 
{
    EngineImplement *impl = new EngineImplement();
    if (!impl->load(file)) 
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

std::shared_ptr<Engine> load(const std::string &file) 
{
    return std::shared_ptr<EngineImplement>((EngineImplement *)loadraw(file));
}

std::string format_shape(const std::vector<int> &shape) 
{
    stringstream output;
    char buf[64];
    const char *fmts[] = {"%d", "x%d"};
    for (int i = 0; i < (int)shape.size(); ++i) 
    {
        snprintf(buf, sizeof(buf), fmts[i != 0], shape[i]);
        output << buf;
    }
    return output.str();
}

}