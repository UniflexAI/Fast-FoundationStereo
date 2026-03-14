#pragma once

#include "NvInfer.h"
#include <string>
#include <vector>

using namespace nvinfer1;

/**
 * TensorRT IPluginV3 plugin for group-wise correlation (GWC) volume.
 *
 * Computes: gwc[b,g,d,h,w] = sum_{c in group g} ref[b,c,h,w] * tgt[b,c,h,w-d]
 *
 * Inputs:  ref (B, C, H, W) fp32 or fp16
 *          tgt (B, C, H, W) fp32 or fp16
 * Output:  gwc (B, G, D, H, W) fp16
 *
 * Attributes: maxdisp (int32), num_groups (int32)
 */
class GwcVolumePlugin : public IPluginV3,
                        public IPluginV3OneCore,
                        public IPluginV3OneBuild,
                        public IPluginV3OneRuntime
{
public:
    GwcVolumePlugin(int maxdisp, int num_groups);
    GwcVolumePlugin() noexcept = delete;
    ~GwcVolumePlugin() noexcept override;

    // IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    IPluginV3* clone() noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept;

    // IPluginV3OneCore
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // IPluginV3OneBuild
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV3OneRuntime
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    int maxdisp_;
    int num_groups_;

private:
    std::string mNamespace;
    PluginFieldCollection mFCToSerialize;
    std::vector<PluginField> mDataToSerialize;

    int B_;
    int C_;
    int H_;
    int W_;
    bool useFp16Input_;
};

class GwcVolumePluginCreator : public IPluginCreatorV3One
{
public:
    GwcVolumePluginCreator();
    ~GwcVolumePluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
};

void launchBuildGwc(DataType type, const void* ref, const void* tgt, void* gwc,
    int B, int C, int H, int W, int D, int G, cudaStream_t stream);
