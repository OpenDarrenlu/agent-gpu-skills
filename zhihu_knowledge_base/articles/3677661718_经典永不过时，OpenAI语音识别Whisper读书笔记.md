# 经典永不过时，OpenAI语音识别Whisper读书笔记

**作者**: 养生的控制人​浙江大学 控制科学与工程博士

**原文链接**: https://zhuanlan.zhihu.com/p/3677661718

---

今天来看一个经典的ASR的工作——Whisper

论文标题：Robust speech recognition via large-scale weak supervision

论文链接：https://arxiv.org/pdf/2212.04356

Whisper 是 OpenAI 在2022年9月推出的一款革命性的自动语音识别（ASR）模型系列。与以往依赖未标注音频数据的模型不同，Whisper 使用了超过68万小时的标注音频进行预训练，此外还包括了11.7万小时的多语种音频，使其能够支持超过96种语言，涵盖了许多数据稀缺的小语种。由于其直接在有监督的语音识别任务上进行预训练，从而能够直接学习语音到文本的映射关系。这种直接的学习方式使得 Whisper 在几乎不需要额外微调的情况下，就已经具备了卓越的 ASR 性能。

Whisper 是一种基于 Transformer 的编码-解码模型。首先，模型将原始音频通过特征提取器转换为对数梅尔频谱图（log-Mel spectrogram）。接着，编码器接收频谱图输入，提取并编码音频的关键特征，生成隐藏状态。解码器则像语言模型一样，基于这些隐藏状态逐步生成对应的文本。

因为模型做的也是下一个token的预测，所以 Whisper 使用交叉熵作为训练目标函数，用于从预定义的词汇表中正确识别和分类目标token。

总的来说，ASR pipeline 可以被拆分为三个环节:

模型输入/预处理：对原始音频输入进行预处理。
模型forward：执行序列到序列的映射。
模型输出/后处理：将模型输出后处理为文本格式。
模型输入/预处理
语音信号的表示和采样：语音信号通过随时间变化的一维数组表示，数组中的数值代表不同时刻的信号幅度。为了便于计算，语音信号通过采样率离散化。采样率越高，越能精确近似连续的语音信号。

采样率匹配的重要性：音频输入的采样率需与模型的采样率匹配，避免出现音频变速或模型性能下降的情况。Whisper特征提取器要求16kHz采样率。

Whisper特征提取器的两步操作：

标准化输入长度：所有音频样本的长度被填充或截断至30秒，这使得无需使用注意力掩码。
生成对数梅尔频谱图：填充后的音频数组被转换为对数梅尔频谱图，这是Whisper模型的输入形式，表示音频的频率和时间分布，符合人类的听觉特性。
def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    将音频数组填充或裁剪到指定的长度（N_SAMPLES），以适应编码器的要求。

    参数:
    array: 音频数组，可以是NumPy数组或PyTorch张量。
    length: 目标长度，默认为N_SAMPLES。
    axis: 操作的轴，默认为最后一个轴（-1）。

    返回:
    处理后的音频数组。
    """
    if torch.is_tensor(array):
        # 如果输入是PyTorch张量
        if array.shape[axis] > length:
            # 如果数组长度大于目标长度，则裁剪
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            # 如果数组长度小于目标长度，则填充
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        # 如果输入是NumPy数组
        if array.shape[axis] > length:
            # 如果数组长度大于目标长度，则裁剪
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            # 如果数组长度小于目标长度，则填充
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    加载用于将STFT投影到Mel频谱图的Mel滤波器矩阵。
    允许解耦librosa依赖；保存使用：

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )

    参数:
    device: 设备（CPU或GPU）。
    n_mels: Mel滤波器的数量，仅支持80或128。

    返回:
    Mel滤波器矩阵，类型为PyTorch张量。
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    计算音频的log-Mel频谱图。

    参数:
    audio: 音频数据，可以是文件路径、NumPy数组或PyTorch张量。
    n_mels: Mel滤波器的数量，默认为80。
    padding: 右侧填充的零样本数。
    device: 如果指定，音频张量将被移动到该设备。

    返回:
    log-Mel频谱图，类型为PyTorch张量，形状为(80, n_frames)。
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            # 如果输入是文件路径，加载音频文件
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        # 将音频张量移动到指定设备
        audio = audio.to(device)
    if padding > 0:
        # 如果需要填充，则在右侧填充零
        audio = F.pad(audio, (0, padding))
    # 创建汉宁窗
    window = torch.hann_window(N_FFT).to(audio.device)
    # 计算STFT（短时傅里叶变换）
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    # 计算幅度平方
    magnitudes = stft[..., :-1].abs() ** 2

    # 获取Mel滤波器矩阵
    filters = mel_filters(audio.device, n_mels)
    # 计算Mel频谱图
    mel_spec = filters @ magnitudes

    # 计算log-Mel频谱图
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

模型forward

模型包含两个主要组件：AudioEncoder 和 TextDecoder。AudioEncoder 通过卷积层和残差注意力块处理音频的梅尔频谱图，生成音频特征表示。TextDecoder 则通过词嵌入和位置嵌入将文本标记转换为向量，并使用包含交叉注意力机制的残差注意力块生成文本的 logits。整个 Whisper 模型通过编码器和解码器的组合，实现了从音频到文本的转换。

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        # 第一个一维卷积层，输入通道数为n_mels，输出通道数为n_state，卷积核大小为3，填充为1
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        # 第二个一维卷积层，输入通道数为n_state，输出通道数为n_state，卷积核大小为3，步幅为2，填充为1
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        # 注册一个缓冲区，用于存储位置编码（positional embedding），形状为(n_ctx, n_state)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        # 创建一个包含多个ResidualAttentionBlock的ModuleList，数量为n_layer
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        # 最后的层归一化层
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        # 对输入进行第一个卷积操作，并使用GELU激活函数
        x = F.gelu(self.conv1(x))
        # 对输入进行第二个卷积操作，并使用GELU激活函数
        x = F.gelu(self.conv2(x))
        # 调整张量的维度顺序，从(batch_size, n_state, n_ctx)变为(batch_size, n_ctx, n_state)
        x = x.permute(0, 2, 1)

        # 检查输入的形状是否与位置编码的形状匹配
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        # 将位置编码加到输入张量上，并转换为与输入张量相同的类型
        x = (x + self.positional_embedding).to(x.dtype)

        # 通过多个ResidualAttentionBlock进行处理
        for block in self.blocks:
            x = block(x)

        # 最后通过层归一化层
        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        # 词嵌入层，将词索引映射到n_state维的向量
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        # 位置嵌入层，形状为(n_ctx, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        # 创建一个包含多个ResidualAttentionBlock的ModuleList，数量为n_layer，每个块包含交叉注意力机制
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        # 最后的层归一化层
        self.ln = LayerNorm(n_state)

        # 创建一个上三角掩码矩阵，用于屏蔽未来的信息
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        # 计算当前的偏移量，用于确定位置嵌入的起始位置
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        # 将词嵌入和位置嵌入相加
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        # 将张量转换为与音频特征相同的类型
        x = x.to(xa.dtype)

        # 通过多个ResidualAttentionBlock进行处理，每个块包含交叉注意力机制
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        # 最后通过层归一化层
        x = self.ln(x)
        # 计算词嵌入的转置与x的点积，得到logits
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        # 保存模型维度信息
        self.dims = dims
        # 初始化音频编码器
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        # 初始化文本解码器
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # 将音频特征输入到编码器中，然后将编码器的输出和文本标记输入到解码器中
        return self.decoder(tokens, self.encoder(mel))

模型输出/后处理
Whisper模型输出：输出为文本标记（text tokens），每个标记对应词汇表中的一个索引。标记器可以将这些标记转换为可读文本字符串。

解码方式：传统ASR使用连接时序分类（CTC）进行解码，但Whisper利用编码器-解码器架构，可以直接使用预训练模型的标记器，无需为每个数据集单独训练CTC标记器。

多语言支持：Whisper的标记器已在96种语言的转录文本上预训练，具备丰富的字节对组合，支持多语言ASR应用。

特殊标记的处理：编码过程中，标记器在序列的开头和结尾添加特殊标记（如转录起止、语言、任务标记等）。在解码时，可以选择忽略这些标记，返回原始输入形式的字符串。


部分核心代码解读，核心跟语言生成模型一样，就是循环预测下一个token

"""
GreedyDecoder 类继承自 TokenDecoder 类，用于实现贪婪解码策略。
"""
class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        """
        初始化方法，设置温度参数和结束标记（eot）。
        
        :param temperature: 温度参数，用于控制采样的随机性。
        :param eot: 结束标记（End of Token），表示序列结束的标记。
        """
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """
        更新方法，根据当前的 logits 生成下一个 token，并更新 tokens 和 sum_logprobs。
        
        :param tokens: 当前的 token 序列。
        :param logits: 当前的 logits（预测的概率分布）。
        :param sum_logprobs: 当前的累积对数概率。
        :return: 更新后的 token 序列和是否完成解码的标志。
        """
        if self.temperature == 0:
            # 如果温度为 0，直接选择概率最大的 token
            next_tokens = logits.argmax(dim=-1)
        else:
            # 否则，根据温度调整后的 logits 进行采样
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        # 计算对数概率
        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        
        # 更新累积对数概率，仅当最后一个 token 不是 eot 时才更新
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        # 如果最后一个 token 是 eot，则将下一个 token 设置为 eot
        next_tokens[tokens[:, -1] == self.eot] = self.eot
        
        # 将新生成的 token 添加到 token 序列中
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        # 判断是否所有序列都已完成解码（即最后一个 token 都是 eot）
        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

def _main_loop(self, audio_features: Tensor, tokens: Tensor):
    """
    主循环方法，用于进行解码的主要逻辑。

    :param audio_features: 音频特征。
    :param tokens: 初始的 token 序列。
    :return: 最终的 token 序列、累积对数概率和 no_speech_probs。
    """
    n_batch = tokens.shape[0]
    sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
    no_speech_probs = [np.nan] * n_batch

    try:
        for i in range(self.sample_len):
            # 获取当前 token 序列的 logits
            logits = self.inference.logits(tokens, audio_features)

            if (
                i == 0 and self.tokenizer.no_speech is not None
            ):  # 保存 no_speech_probs
                probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

            # 只考虑最后一个 token 的 logits
            logits = logits[:, -1]

            # 应用 logit 过滤器，例如抑制某些 token 或施加惩罚
            for logit_filter in self.logit_filters:
                logit_filter.apply(logits, tokens)

            # 更新 token 序列和累积对数概率
            tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

            # 如果解码完成或 token 序列长度超过最大长度，则退出循环
            if completed or tokens.shape[-1] > self.n_ctx:
                break
    finally:
        # 清理缓存
        self.inference.cleanup_caching()

    return tokens, sum_logprobs, no_speech_probs


参考资料

https://arxiv.org/pdf/2212.04356
https://huggingface.co/blog/fine-tune-whisper
https://github.com/openai/whisper/tree/main
