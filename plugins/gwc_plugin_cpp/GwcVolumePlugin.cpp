#include "GwcVolumePlugin.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>

char const* gGwcVolumePluginName{"BuildGwcVolume"};
char const* gGwcVolumePluginVersion{"1"};

GwcVolumePlugin::GwcVolumePlugin(int maxdisp, int num_groups)
    : maxdisp_(maxdisp)
    , num_groups_(num_groups)
    , B_(1)
    , C_(0)
    , H_(0)
    , W_(0)
    , useFp16Input_(false)
{
}

GwcVolumePlugin::~GwcVolumePlugin() noexcept {}

IPluginCapability* GwcVolumePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try {
        if (type == PluginCapabilityType::kBUILD)
            return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME)
            return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) {}
    return nullptr;
}

IPluginV3* GwcVolumePlugin::clone() noexcept
{
    try {
        return new GwcVolumePlugin(maxdisp_, num_groups_);
    } catch (...) {}
    return nullptr;
}

char const* GwcVolumePlugin::getPluginName() const noexcept { return gGwcVolumePluginName; }
char const* GwcVolumePlugin::getPluginVersion() const noexcept { return gGwcVolumePluginVersion; }
char const* GwcVolumePlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

void GwcVolumePlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

int32_t GwcVolumePlugin::getNbOutputs() const noexcept { return 1; }

int32_t GwcVolumePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    (void)nbInputs;
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t GwcVolumePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    (void)shapeInputs;
    (void)nbShapeInputs;
    outputs[0].nbDims = 5;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = exprBuilder.constant(num_groups_);
    outputs[0].d[2] = exprBuilder.constant(maxdisp_);
    outputs[0].d[3] = inputs[0].d[2];
    outputs[0].d[4] = inputs[0].d[3];
    return 0;
}

bool GwcVolumePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut[pos].desc.format != TensorFormat::kLINEAR) {
        return false;
    }

    // Inputs: ref, tgt. Require same type for both inputs because enqueue()
    // dispatches one kernel path based on input 0 type and interprets both
    // input pointers with that type.
    if (pos == 0) {
        return inOut[0].desc.type == DataType::kFLOAT || inOut[0].desc.type == DataType::kHALF;
    }
    if (pos == 1) {
        return inOut[1].desc.type == inOut[0].desc.type;
    }

    // Output is always fp16 cost volume.
    if (pos == 2) {
        return inOut[2].desc.type == DataType::kHALF;
    }
    return false;
}

int32_t GwcVolumePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    (void)out;
    (void)nbOutputs;
    useFp16Input_ = (in[0].desc.type == DataType::kHALF);
    return 0;
}

int32_t GwcVolumePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    (void)nbInputs;
    B_ = in[0].dims.d[0];
    C_ = in[0].dims.d[1];
    H_ = in[0].dims.d[2];
    W_ = in[0].dims.d[3];
    useFp16Input_ = (in[0].type == DataType::kHALF);

    // IMPORTANT: derive (G, D) from the output tensor shape to avoid
    // relying on plugin field parsing (which can differ across TRT/ONNX parser versions).
    // Output is (B, G, D, H, W).
    if (nbOutputs >= 1 && out != nullptr && out[0].dims.nbDims == 5) {
        num_groups_ = out[0].dims.d[1];
        maxdisp_ = out[0].dims.d[2];
    }
    return 0;
}

size_t GwcVolumePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    (void)inputs;
    (void)nbInputs;
    (void)outputs;
    (void)nbOutputs;
    return 0;
}

IPluginV3* GwcVolumePlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    (void)context;
    return clone();
}

PluginFieldCollection const* GwcVolumePlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("maxdisp", &maxdisp_, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("num_groups", &num_groups_, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

int32_t GwcVolumePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    (void)workspace;
    DataType type = inputDesc[0].type;
    if (inputDesc[1].type != type) {
        return 1;
    }

    int B = B_, C = C_, H = H_, W = W_, D = maxdisp_, G = num_groups_;
    if (inputDesc[0].dims.nbDims == 4) {
        B = inputDesc[0].dims.d[0];
        C = inputDesc[0].dims.d[1];
        H = inputDesc[0].dims.d[2];
        W = inputDesc[0].dims.d[3];
    }
    if (outputDesc[0].dims.nbDims == 5) {
        G = outputDesc[0].dims.d[1];
        D = outputDesc[0].dims.d[2];
    }
    if (B <= 0 || C <= 0 || H <= 0 || W <= 0 || D <= 0 || G <= 0 || (C % G) != 0) {
        return 1;
    }

    launchBuildGwc(type, inputs[0], inputs[1], outputs[0],
        B, C, H, W, D, G, stream);
    return 0;
}

// ---------------------------------------------------------------------------
// Plugin Creator
// ---------------------------------------------------------------------------

GwcVolumePluginCreator::GwcVolumePluginCreator()
{
    static int32_t placeholder = 0;
    mPluginAttributes.emplace_back("maxdisp", &placeholder, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("num_groups", &placeholder, PluginFieldType::kINT32, 1);
    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

char const* GwcVolumePluginCreator::getPluginName() const noexcept { return gGwcVolumePluginName; }
char const* GwcVolumePluginCreator::getPluginVersion() const noexcept { return gGwcVolumePluginVersion; }
PluginFieldCollection const* GwcVolumePluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV3* GwcVolumePluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    (void)name;
    (void)phase;
    int32_t maxdisp = 48;
    int32_t num_groups = 8;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        // Depending on parser/version, fields may arrive either as "maxdisp"
        // or as the ONNX attribute-style "maxdisp_i". Accept both.
        if (strcmp(fc->fields[i].name, "maxdisp") == 0 || strcmp(fc->fields[i].name, "maxdisp_i") == 0)
            maxdisp = *static_cast<int32_t const*>(fc->fields[i].data);
        else if (strcmp(fc->fields[i].name, "num_groups") == 0 || strcmp(fc->fields[i].name, "num_groups_i") == 0)
            num_groups = *static_cast<int32_t const*>(fc->fields[i].data);
    }
    return new GwcVolumePlugin(maxdisp, num_groups);
}

void GwcVolumePluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

char const* GwcVolumePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(GwcVolumePluginCreator);
