# Let's import these libraries
import torch  # PyTorch, the backend for transformers
from transformers import AutoModelForCausalLM
from transformers import TextStreamer, TextIteratorStreamer
from transformers import AutoTokenizer
from threading import Thread
from pdf_reader import PdfReader
from local_embedding import LocalEmbedding
import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


class AiModel():

    # Use a pre-quantized Hub checkpoint by default (auto-downloads + caches).
    # unsloth/Qwen2.5-3B-Instruct-bnb-4bit is already quantized to 4-bit,
    # so it downloads only ~2GB (vs ~6GB for the fp16 original) and never
    # needs the full fp16 weights loaded into memory before quantizing —
    # important on machines with limited RAM (e.g. 8GB laptops), where
    # loading the full fp16 checkpoint can trigger Windows paging-file
    # errors even though the final quantized model fits fine in VRAM.
    def __init__(self, model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit"):
        '''
            initializing my AiModel class where we need the model name to create a tokenizer and a model
            the tokenizer will transform our text into numbers for our model to understand then will transform the numbers from the model to text so we understand it
            the model is the LLM that will think and give us the answers to our questions
        '''
        self.model_name = model_name
        print("running checks to make sure everything is good...")
        self.hugging_face_auth()
        self.hardware_check()
        print("we are creating the model this might take a while please wait...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            print("[LocalLLM] Falling back to explicit LlamaTokenizer...")
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        # This checkpoint is already 4-bit quantized — its config.json declares
        # the quantization scheme, so from_pretrained picks it up automatically.
        # bitsandbytes still needs to be installed (it's what decodes the
        # quantized weights at load time), but we don't build our own
        # BitsAndBytesConfig here to avoid conflicting with the one baked in.
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            device_map="auto",
        )
    

    def hardware_check(self):
        '''
            making sure we are working on a local GPU rather than CPU to take advantage of Local LLMs
        '''
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: No GPU detected. Generation will be slow, and a 7B+ model may not fit in RAM at all.")
    

    def hugging_face_auth(self):
        '''
            in order to download the right model to work on some of the model are gated by HuggingFace therefore we must authenticate first.

            Reads HF_TOKEN from the environment (set it in your .env file as:
                HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
            and make sure main.py calls load_dotenv() before this runs).

            Only gated models (like Mistral) actually require a valid token.
            Ungated models (like Qwen2.5) will work fine even with no token set,
            so we don't hard-fail here — we just warn and let from_pretrained
            surface a clear error later if auth turns out to be required.
        '''
        hugging_face_token = os.environ.get("HF_TOKEN")

        if not hugging_face_token:
            print("[LocalLLM] WARNING: HF_TOKEN not found in environment. "
                  "This is fine for ungated models, but will fail for gated "
                  "models (e.g. Mistral) or private repos.")
            return

        print("Attempting Hugging Face login...")
        try:
            login(token=hugging_face_token)
            print("Login successful!")
        except Exception as e:
            print(f"[LocalLLM] WARNING: Hugging Face login failed: {e}")


    def ask_a_question(self, prompt="Hello there!"):
        '''
            formats the prompt as a chat message so the instruction-tuned model
            knows to respond rather than do raw text completion
        '''
        # wrap the prompt in the chat format the model was trained on
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # getting input and prompt
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        # Streaming output — tokens are printed to the terminal as they are generated,
        # instead of waiting for the full response to be built in memory first.
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # max_new_tokens caps the response length; streamer handles all printing internally.
        self.model.generate(**inputs, max_new_tokens=1000, streamer=streamer)
    
    def ask_a_question_from_pdf(self, pdf_path, prompt="tell me what is this pdf about"):
        '''
            this function allows the user to take a pdf and ask some questions about the pdf, 
            performing RAG operation
        '''
        # creating our PdfReader to work with the pdf text
        pdf_reader = PdfReader(pdf_path)
        pdf_paragraphs = pdf_reader.get_paragraphs()
        
        # embedding and indexing all chunks
        local_embedding = LocalEmbedding()
        local_embedding.build_index(pdf_paragraphs)

        # getting relevant sections of the pdf
        relevent_sections = local_embedding.get_context(prompt, 10)

        # crafting message
        full_prompt_for_rag = self.full_prompt_for_rag(relevent_sections=relevent_sections, question_prompt=prompt)
        messages = [{"role": "user", "content": full_prompt_for_rag}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # getting input and prompt
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        # Streaming output — tokens are printed to the terminal as they are generated,
        # instead of waiting for the full response to be built in memory first.
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # max_new_tokens caps the response length; streamer handles all printing internally.
        self.model.generate(**inputs, max_new_tokens=1000, streamer=streamer)


    def ask_a_question_from_pdf_stream(self, pdf_path: str, prompt: str = "tell me what is this pdf about", local_embedding=None):
        '''
            Streaming variant of ask_a_question_from_pdf.
            Yields decoded text chunks via TextIteratorStreamer so callers (e.g. st.write_stream)
            can consume tokens in real time without blocking on stdout.

            Args:
                pdf_path:        Path to the PDF on disk.
                prompt:          User question string.
                local_embedding: Pre-built LocalEmbedding instance (already indexed).
                                 If None, builds the index from scratch.
            Yields:
                str chunks as the model generates them.
        '''
        if local_embedding is None:
            pdf_reader = PdfReader(pdf_path)
            pdf_paragraphs = pdf_reader.get_paragraphs()
            local_embedding = LocalEmbedding()
            local_embedding.build_index(pdf_paragraphs)

        relevant_sections = local_embedding.get_context(prompt, k=10)
        full_prompt = self.full_prompt_for_rag(
            relevent_sections=relevant_sections,
            question_prompt=prompt,
        )
        messages = [{"role": "user", "content": full_prompt}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        # TextIteratorStreamer stores tokens in a Queue instead of printing to stdout.
        # timeout=30 prevents blocking forever if the generation thread crashes.
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30.0
        )

        # model.generate() is blocking — run it in a daemon thread so the main
        # thread can iterate the streamer queue without deadlocking.
        thread = Thread(
            target=self.model.generate,
            kwargs=dict(**inputs, max_new_tokens=1000, streamer=streamer),
            daemon=True,
        )
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()


    def full_prompt_for_rag(self, relevent_sections, question_prompt):
        '''
            this is a prompt constructor that will put together the user question, the pdf relevant sections, and system prompt
        '''
        return f"""
            <|system|>
                You are an AI assistant. Answer the following question based *only* on the provided document text. 
                If the answer is not found in the document, say "The document does not contain information on this topic." Do not use any prior knowledge.

                Document Text:
                ---
                    {relevent_sections}
                ---
            <|end|>
            |user|>
                Question: {question_prompt}
            <|end|>
            <|assistant|>
                Answer:
    """


#new_ai_model = AiModel()
#new_ai_model.ask_a_question_from_pdf("./pdfs/2025-q1-earnings-transcript.pdf")