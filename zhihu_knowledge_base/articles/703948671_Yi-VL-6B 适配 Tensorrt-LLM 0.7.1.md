# Yi-VL-6B 适配 Tensorrt-LLM 0.7.1

**作者**: JiLi-QA百无一用

**原文链接**: https://zhuanlan.zhihu.com/p/703948671

---

​
目录
收起
1. docker 环境
2. 模型分析
2.1 首先是视觉模型起作用
2.2 输入 llm
3. 视觉模型导出 onnx && build engine
3.1 导出的核心代码
4. LLM 部分 build engine
5. 运行pipeline
5.1 trt 推理结果

Yi-VL 结构和 llava近乎一致，适配流程参考 tensorrt-llm 0.9.0 的 llava 模型（TensorRT-LLM/examples/multimodal）
首先是将视觉部分导出为 onnx，然后build engine
然后将语言部分直接用 llama的代码即可以 build engine
1. docker 环境
构造docker
<https://github.com/NVIDIA/TensorRT-LLM.git>
git checkout tags/v0.7.1
docker build --pull \
			--target devel\
			--file docker/Dockerfile.multi \
			--tag%20tensorrt_llm/devel:0.7.1 .
编译 tensorrt-llm
docker run -itd --name xxxxx  --gpus all -v $PWD:/workspace -w /workspace -v /etc/localtime:/etc/localtime --net=host --shm-size=64g --privileged tensorrt_llm/devel:0.7.1 bash 

docker exec -it xxxxx bash

git clone <https://github.com/NVIDIA/TensorRT-LLM.git>
git checkout tags/v0.7.1
# 安装 tensorrt_llm 
python3 scripts/build_wheel.py --clean  --trt_root /usr/local/tensorrt

export PIP_EXTRA_INDEX_URL='<https://pypi.nvidia.com>'
pip3 install ./build/tensorrt_llm*.whl
2. 模型分析
直接打印模型结构会造成巨大的误解：以为是 llama + vision_tower
class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        config._flash_attn_2_enabled = True  ######set flash attention2!!!!!!
        super(LlavaLlamaModel, self).__init__(config)

self.model = LlavaLlamaModel(config)

self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
2.1 首先是视觉模型起作用
(            
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
  )
2.2 输入 llm
outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
所以其真实结构，与llava 近乎一致，有一些微小的变化
3. 视觉模型导出 onnx && build engine
python export_Yi_onnx.py \
    --model_type llava \
    --model_path /workspace/Yi-VL-6B \
    --output_dir /workspace/Yi-VL-engine
获取模型的三部分参数信息
	llava-1.5-7B	Yi-VL-6b
vision_tower	model.vision_tower	model.get_vision_tower()
mm_projector	model.multi_modal_projector	model.get_model().mm_projector
vision_feature_layer	model.config.vision_feature_layer	model.config.mm_vision_select_layer
3.1 导出的核心代码
def build_llava_engine(args):
    key_info["model_path"] = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16,
    )
    image_processor = None
    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()

    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor

    mm_projector = model.get_model().mm_projector
    processor = image_processor

    raw_image = Image.new('RGB', [10, 10])  # dummy image
    image = processor(text="dummy", images=raw_image,
                      return_tensors="pt")['pixel_values'].to(
                          args.device, torch.float16)

    class LlavaVisionWrapper(torch.nn.Module):

        def __init__(self, tower, projector, feature_layer):
            super().__init__()
            self.tower = tower
            self.projector = projector
            self.feature_layer = feature_layer

        def forward(self, image):
            # all_hidden_states = self.tower(
            #     image, output_hidden_states=True).hidden_states
            image_features = self.tower(image)
            features = image_features
            return self.projector(features)

    # vision_tower = model.get_vision_tower()
    wrapper = LlavaVisionWrapper(vision_tower.to(args.device),
                                 mm_projector.to(args.device),
                                 model.config.mm_vision_select_layer)

    export_visual_wrapper_onnx(wrapper, image, args.output_dir)
    build_trt_engine(image.shape[2], image.shape[3], args.output_dir,
                     args.max_batch_size)
4. LLM 部分 build engine
cd TensorRT-LLM/examples/llama 中进行编译
python build.py \
        --model_dir /workspace/Yi-VL-6B \
        --output_dir /workspace/Yi-VL-engine/${MODEL_NAME}/fp16/1-gpu \
        --use_gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_output_len 512 \
        --max_prompt_embedding_table_size 1024 > log_build_Yi.log 2>&1
打印结构确认与 hf 模型一致
5. 运行pipeline
5.1 trt 推理结果
run.py 参考 0.9.0 的 TensorRT-LLM/examples/multimodal/run.py
python run.py \
        --max_new_tokens 30 \
        --hf_model_dir /workspace/Yi-VL-6B \
        --visual_engine_dir /workspace/Yi-VL-engine \
        --llm_engine_dir /workspace/Yi-VL-engine/fp16/1-gpu \
        --decoder_llm \
        --input_text "Question: which city is this? Answer:"
