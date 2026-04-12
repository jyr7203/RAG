import os
import torch
import importlib.metadata as _meta
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from config import Config
from logger_setting import get_logger

log = get_logger("ModelLoader")

_hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
if _hf_token:
    login(token=_hf_token)
    log.info("HuggingFace 로그인 완료")

def _bnb_available() -> bool:
    """bitsandbytes가 실제로 설치돼 있는지 패키지 메타데이터로 확인합니다."""
    try:
        _meta.version("bitsandbytes")
        return True
    except _meta.PackageNotFoundError:
        return False


class KananaModel:
    _model     = None
    _tokenizer = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            log.info(f"Kanana 모델 로드 중... ({Config.LLM_MODEL})")

            cls._tokenizer = AutoTokenizer.from_pretrained(
                Config.LLM_MODEL,
                trust_remote_code=True,
                use_fast=True,
            )
            # pad_token 미설정 시 eos_token으로 대체
            if cls._tokenizer.pad_token_id is None:
                cls._tokenizer.pad_token = cls._tokenizer.eos_token

            use_cuda = (Config.DEVICE == "cuda")
            use_bnb  = use_cuda and _bnb_available()

            if use_bnb:
                # GPU 세대 확인 (A100=8.0 이상 → bfloat16, T4=7.5 이하 → float16)
                major, _ = torch.cuda.get_device_capability()
                compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
                log.info(
                    f"CUDA + bitsandbytes 감지: 4bit 양자화 모드로 로드합니다. "
                    f"(GPU compute capability: {major}.x → compute_dtype={compute_dtype})"
                )
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                load_kwargs = {
                    "quantization_config": bnb_config,
                    "device_map": "auto",
                }

            elif use_cuda:
                # bitsandbytes 없을 때 — GPU 세대에 맞는 dtype으로 full precision 로드
                major, _ = torch.cuda.get_device_capability()
                dtype = torch.bfloat16 if major >= 8 else torch.float16
                log.warning(
                    f"CUDA 있음 / bitsandbytes 없음: {dtype} 전체 정밀도로 로드합니다. "
                    f"(GPU compute capability: {major}.x)"
                )
                load_kwargs = {
                    "torch_dtype": dtype,
                    "device_map": "auto",
                }

            else:
                # CPU — float32가 float16보다 안정적
                log.warning("CUDA 없음: CPU + float32 모드로 로드합니다. (속도 저하 예상)")
                torch.set_num_threads(os.cpu_count())
                load_kwargs = {
                    "torch_dtype": torch.float32,
                    "device_map": "auto",
                }

            try:
                cls._model = AutoModelForCausalLM.from_pretrained(
                    Config.LLM_MODEL,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    **load_kwargs
                ).eval()
                
                # GPU 정보 로그
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    log.info(f"모델 로드 완료! GPU: {gpu_name} ({gpu_mem:.1f}GB)")
                else:
                    log.info("모델 로드 완료! (CPU 모드)")
            except Exception as e:
                log.error(f"모델 로드 실패: {e}")
                cls._model     = None
                cls._tokenizer = None
                raise

        return cls._model, cls._tokenizer


if __name__ == "__main__":
    model, tokenizer = KananaModel.get_model()
    print("출력 테스트: 로드가 성공했습니다.")

