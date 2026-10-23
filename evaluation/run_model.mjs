// Real local ONNX inference using the exact website retrieval and prompt functions.
// This is not a browser-GPU run; device and quantization are recorded explicitly.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
import {createRequire} from 'node:module';
import {prepareRag,assess} from '../lib/rag.ts';
const base=process.env.QAIRU_TOOLKIT_NODE;
const require=createRequire(resolve(base||'.','package.json'));
const {pipeline,env}=require('@huggingface/transformers');
const model='onnx-community/Qwen2.5-0.5B-Instruct',revision='516c8d04add8a80c5228f32102b57953b8d421a9';
env.cacheDir=process.env.QAIRU_MODEL_CACHE||'/tmp/qairu-model-cache';
const g=await pipeline('text-generation',model,{device:'cpu',dtype:'q4',revision,session_options:{intraOpNumThreads:4,interOpNumThreads:1},progress_callback:p=>{if(p.status==='done')console.log('Loaded',p.file);}});
const cases=JSON.parse(readFileSync(new URL('../public/evaluations/cases.json',import.meta.url)));
const run_id=crypto.randomUUID(),records=[];
for(const c of cases)for(const mode of ['baseline','protected']){
 const p=prepareRag(c,mode),start=performance.now();
 const rendered=g.tokenizer.apply_chat_template(p.messages,{tokenize:false,add_generation_prompt:true});
 const input_tokens=Number(g.tokenizer(rendered).input_ids.dims.at(-1));
 const result=await g(p.messages,{max_new_tokens:64,do_sample:false});
 const text=result[0].generated_text;const raw_output=typeof text==='string'?text:text.at(-1).content;
 records.push({uuid:crypto.randomUUID(),run_id,timestamp_utc:new Date().toISOString(),case:c,mode,query:c.query,contexts:p.contexts,messages:p.messages,retrieved_context_ids:p.retrieved.map(d=>d.id),reference_context_ids:c.reference_ids,raw_output,...assess(c,raw_output),model_generation_seconds:(performance.now()-start)/1000,input_tokens,model_id:model,revision,device:'cpu',dtype:'q4',max_new_tokens:64,do_sample:false,runtime:'Transformers.js 3.8.1 / ONNX Runtime Node',inference:'live_local_onnx_qwen',target:'QAIRU synthetic RAG v1',scope:'Shared website source; local CPU inference, not browser GPU or corporate RAG'});
 writeFileSync(new URL('../public/evaluations/live-results.json',import.meta.url),JSON.stringify({run_id,records},null,2));
 console.log(c.id,mode,JSON.stringify(raw_output));
}
