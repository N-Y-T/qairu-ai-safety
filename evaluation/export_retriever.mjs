import {DOCUMENTS,FEATURES,retrieve,vectorize} from '../lib/rag.ts';
import {writeFileSync} from 'node:fs';
const queries=['leave days','expenses receipts','vpn portal','office hours','leave expenses','cafeteria'];
writeFileSync(new URL('../public/evaluations/retriever.json',import.meta.url),JSON.stringify({features:FEATURES,documents:[...DOCUMENTS].sort((a,b)=>a.id.localeCompare(b.id)),probes:queries.map(query=>({query,vector:vectorize(query),retrieved:retrieve(query).map(d=>({id:d.id,score:d.score}))}))},null,2));
