// A stand-in for the production front door's routing (deploy/vps/sites/00-audio-cleaner-snippets.caddy) for the browser suites:
// the live WebSocket (/api/stream/ws) goes to the live process (acw.live_app), everything else to the main app.
//   node tests/front.mjs <port> <main-app-port> <live-app-port>
import { createServer, request } from 'node:http';
import { connect } from 'node:net';

const [port, mainPort, livePort] = process.argv.slice(2).map(Number);
const target = (url) => (url.split('?')[0] === '/api/stream/ws' ? livePort : mainPort);

const server = createServer((req, res) => {
  const up = request({ host: '127.0.0.1', port: target(req.url), path: req.url, method: req.method, headers: req.headers }, (r) => { res.writeHead(r.statusCode, r.headers); r.pipe(res); });
  up.on('error', () => { res.writeHead(502); res.end('Bad Gateway'); });
  req.pipe(up);
});
server.on('upgrade', (req, socket, head) => {
  const up = connect(target(req.url), '127.0.0.1', () => {
    up.write(`${req.method} ${req.url} HTTP/1.1\r\n${req.rawHeaders.reduce((s, v, i) => s + (i % 2 ? `: ${v}\r\n` : v), '')}\r\n`);
    up.write(head);
    socket.pipe(up).pipe(socket);
  });
  up.on('error', () => socket.destroy());
  socket.on('error', () => up.destroy());
});
server.listen(port, '127.0.0.1');
