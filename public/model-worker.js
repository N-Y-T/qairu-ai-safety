// Real ONNX model inference. No canned model replies or attack matching.
const MODEL = 'onnx-community/Qwen2.5-0.5B-Instruct';
const REVISION = '516c8d04add8a80c5228f32102b57953b8d421a9';
let generator, device, dtype, loading = false;
self.onmessage = async ({data}) => {
  try {
    if (data.type === 'load') {
      if (loading) return;
      loading = true;
      self.postMessage({type:'status', text:'Подключаю библиотеку модели…'});
      const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js');
      env.allowLocalModels = false;
      env.backends.onnx.wasm.numThreads = 1;
      let adapter;
      if (data.device !== 'wasm') { try { adapter = await navigator.gpu?.requestAdapter(); } catch {} }
      device = adapter ? 'webgpu' : 'wasm';
      dtype = adapter ? (adapter.features.has('shader-f16') ? 'q4f16' : 'q4') : 'q8';
      self.postMessage({type:'status',text:device==='webgpu'?'Загружаю Qwen для GPU…':'Загружаю Qwen для CPU…'});
      generator = await pipeline('text-generation', MODEL, {
        device, dtype, revision: REVISION,
        progress_callback: p => { if (p.status==='progress') self.postMessage({type:'progress', file:p.file, loaded:p.loaded, total:p.total, progress:p.progress}); else if (p.status==='done') self.postMessage({type:'status', text:'Подготавливаю '+p.file+'…'}); }
      });
      self.postMessage({type:'ready',device,dtype,model_id:MODEL,revision:REVISION});
    } else if (data.type === 'generate') {
      if (!generator) throw new Error('Сначала запусти модель.');
      const rendered = generator.tokenizer.apply_chat_template(data.messages,{tokenize:false,add_generation_prompt:true});
      const encoded = generator.tokenizer(rendered);
      const inputTokens = encoded.input_ids.dims.at(-1);
      if (inputTokens > 2048) throw new Error('Запрос превышает лимит 2048 токенов. Сократи сообщение.');
      const start = performance.now();
      const result = await generator(data.messages,{max_new_tokens:64,do_sample:false});
      const generated = result[0].generated_text;
      const raw = typeof generated === 'string' ? generated : generated.at(-1).content;
      self.postMessage({type:'result',id:data.id,raw_output:raw,input_tokens:inputTokens,model_generation_seconds:(performance.now()-start)/1000,model_id:MODEL,revision:REVISION,device,dtype,max_new_tokens:64,do_sample:false,runtime:'Transformers.js 3.8.1 / ONNX Runtime'});
    }
  } catch (error) { loading=false; self.postMessage({type:'error',id:data.id,message:String(error?.message || error)}); }
};
