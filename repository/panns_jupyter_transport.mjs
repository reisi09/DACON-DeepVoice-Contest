import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {randomUUID} from 'node:crypto';

const workspace = 'wsp-nvxqw1jf9zdv';
const codes = new Set(['ENOENT', 'EACCES', 'EPERM', 'EMFILE', 'ENFILE', 'ENOSPC', 'ETIMEDOUT',
  'ECONNRESET', 'ECONNREFUSED', 'ENOTFOUND', 'EAI_AGAIN', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_SOCKET']);
let phase = 'payload_read';

function diagnostic(error, location) {
  const name = ['Error', 'TypeError', 'SyntaxError', 'TimeoutError', 'AbortError'].includes(error?.name) ? error.name : 'UnknownError';
  const nativeCode = error?.code ?? error?.cause?.code;
  const kind = ['timeout', 'network_error', 'identity_error', 'http_error', 'protocol_error', 'remote_error', 'websocket_error'].includes(error?.transportKind)
    ? error.transportKind : (name === 'TimeoutError' || name === 'AbortError' || nativeCode === 'ETIMEDOUT'
    ? 'timeout' : name === 'TypeError' && error.message === 'fetch failed' ? 'network_error' : name);
  return {phase: location, kind, code: codes.has(nativeCode) ? nativeCode : null,
    http_status: Number.isInteger(error?.httpStatus) ? error.httpStatus : null};
}

function failure(kind, httpStatus = null) {
  return Object.assign(new Error(kind), {transportKind: kind, httpStatus});
}

try {
  const code = readFileSync(process.argv[2], 'utf8');
  phase = 'workspace_lookup';
  const detail = JSON.parse(execFileSync('/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/bin/vesslctl',
    ['--org', 'qwdkw', 'workspace', 'show', workspace, '-o', 'json'],
    {encoding: 'utf8', timeout: 10000, maxBuffer: 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe']}));
  phase = 'workspace_identity';
  const endpoints = detail.endpoints.filter(item => item.name === 'jupyter');
  if (detail.workspace.slug !== workspace || endpoints.length !== 1) throw failure('identity_error');
  const endpoint = new URL(endpoints[0].httpUrl);
  const token = endpoint.searchParams.get('token');
  if (endpoint.protocol !== 'https:' || endpoint.hostname !== 'jupyter-' + workspace + '.betelgeuse.cloud.vessl.ai'
      || endpoint.username || endpoint.password || (endpoint.port && endpoint.port !== '443') || !token) throw failure('identity_error');
  const base = endpoint.origin;
  const headers = {Authorization: 'token ' + token, 'Content-Type': 'application/json'};
  phase = 'kernel_create';
  const response = await fetch(base + '/api/kernels', {method: 'POST', headers,
    body: JSON.stringify({name: 'python3'}), signal: AbortSignal.timeout(15000)});
  if (!response.ok) throw failure('http_error', response.status);
  const kernel = await response.json();
  if (typeof kernel.id !== 'string' || !/^[a-zA-Z0-9-]+$/.test(kernel.id)) throw failure('protocol_error');
  let executed = false;
  try {
    phase = 'execute';
    const url = new URL(base.replace('https:', 'wss:') + '/api/kernels/' + kernel.id + '/channels');
    url.searchParams.set('token', token);
    const socket = new WebSocket(url);
    const id = randomUUID();
    try {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(failure('timeout')), 40000);
        const fail = error => { clearTimeout(timer); reject(error); };
        let reply = null, idle = false;
        socket.onopen = () => {
          try {
            socket.send(JSON.stringify({header: {msg_id: id, username: 'codex', session: randomUUID(),
              msg_type: 'execute_request', version: '5.3'}, parent_header: {}, metadata: {}, channel: 'shell',
              content: {code, silent: false, store_history: false, user_expressions: {}, allow_stdin: false, stop_on_error: true}}));
          } catch { fail(failure('websocket_error')); }
        };
        socket.onmessage = event => {
          try {
            const message = JSON.parse(event.data);
            if (message.parent_header?.msg_id !== id) return;
            const type = message.header?.msg_type ?? message.msg_type;
            if (type === 'stream') process.stdout.write(message.content.text);
            if (type === 'error') process.stdout.write('{"error":"RemoteExecutionError"}\n');
            if (message.channel === 'shell' && type === 'execute_reply') reply = message.content.status;
            if (message.channel === 'iopub' && type === 'status' && message.content.execution_state === 'idle') idle = true;
            if (reply !== null && idle) {
              clearTimeout(timer);
              if (reply === 'ok') resolve(); else reject(failure('remote_error'));
            }
          } catch { fail(failure('protocol_error')); }
        };
        socket.onerror = () => fail(failure('websocket_error'));
      });
      executed = true;
    } finally { socket.close(); }
  } finally {
    try {
      const cleanup = await fetch(base + '/api/kernels/' + kernel.id,
        {method: 'DELETE', headers, signal: AbortSignal.timeout(5000)});
      if (!cleanup.ok && cleanup.status !== 404) throw failure('http_error', cleanup.status);
    } catch (error) {
      process.stderr.write('PANNS_TRANSPORT_WARNING ' + JSON.stringify(diagnostic(error, 'kernel_cleanup')) + '\n');
      if (!executed) phase = 'execute';
    }
  }
} catch (error) {
  process.stderr.write('PANNS_TRANSPORT_ERROR ' + JSON.stringify(diagnostic(error, phase)) + '\n');
  process.exitCode = 1;
}
