/**
 * 本地 HTTPS 证书生成（用 Git 自带的 openssl，不引入 npm 依赖）。
 *
 * 为什么需要它：**WebGPU 只在安全上下文里可用**。
 * 手机上访问 `http://192.168.x.x:5173` 不是安全上下文，`navigator.gpu` 直接不存在。
 * 想用 HTTPS 就得有个证书；公开 CA 签不了内网 IP，只能自己签一个本地 CA，
 * 让手机信任它一次。
 *
 * 产出（放在 node_modules/.cache/certs/，不入库）：
 *   ca.pem       —— 拷到手机上安装（CA 证书）
 *   server.pem   —— vite 用的叶证书
 *   server.key   —— 叶证书私钥
 */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';

/** Git for Windows 自带的 openssl；也允许用 PATH 上或 OPENSSL_BIN 指定的。 */
export function findOpenssl ()
{
  if ( process.env.OPENSSL_BIN && fs.existsSync( process.env.OPENSSL_BIN ) ) return process.env.OPENSSL_BIN;
  const cands = [
    'C:\\Program Files\\Git\\usr\\bin\\openssl.exe',
    'C:\\Program Files\\Git\\mingw64\\bin\\openssl.exe',
    '/usr/bin/openssl',
    '/usr/local/bin/openssl',
  ];
  for ( const c of cands ) if ( fs.existsSync( c ) ) return c;
  // 退一步：交给 PATH
  const probe = spawnSync( 'openssl', [ 'version' ], { encoding: 'utf8' } );
  return probe.status === 0 ? 'openssl' : null;
}

/** 去掉 IPv4-mapped IPv6 前缀（Node 有时会给出 ::ffff:192.168.1.5）。 */
export function cleanIps ( ips )
{
  return [ ...new Set(
    ips
      .map( ( ip ) => ip.replace( /^::ffff:/, '' ) )
      .filter( ( ip ) => /^\d+\.\d+\.\d+\.\d+$/.test( ip ) ),
  ) ];
}

function run ( openssl, args, cwd )
{
  const r = spawnSync( openssl, args, { cwd, encoding: 'utf8' } );
  if ( r.status !== 0 )
  {
    throw new Error( `openssl ${ args[ 0 ] } 失败（exit ${ r.status }）：${ ( r.stderr || r.stdout || '' ).trim() }` );
  }
}

/**
 * 确保存在一套能覆盖当前 IP 的证书。
 * 返回 { dir, ca, cert, key, openssl, regenerated, ips }。
 */
export function ensureCerts ( { dir, ips } )
{
  const openssl = findOpenssl();
  if ( !openssl ) throw new Error( '找不到 openssl：装 Git for Windows，或用 OPENSSL_BIN 指定路径' );
  const clean = cleanIps( ips );
  const stamp = [ 'localhost', '127.0.0.1', ...clean ].sort().join( ',' );

  const caKey = path.join( dir, 'ca.key' );
  const caPem = path.join( dir, 'ca.pem' );
  const srvKey = path.join( dir, 'server.key' );
  const srvCsr = path.join( dir, 'server.csr' );
  const srvPem = path.join( dir, 'server.pem' );
  const stampFile = path.join( dir, 'san.txt' );

  fs.mkdirSync( dir, { recursive: true } );
  const upToDate = fs.existsSync( stampFile ) && fs.readFileSync( stampFile, 'utf8' ) === stamp &&
    fs.existsSync( caPem ) && fs.existsSync( srvKey ) && fs.existsSync( srvPem );
  if ( upToDate )
  {
    return { dir, ca: caPem, cert: srvPem, key: srvKey, openssl, regenerated: false, ips: clean };
  }

  // 1) 本地 CA（10 年）
  run( openssl, [
    'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
    '-keyout', 'ca.key', '-out', 'ca.pem', '-days', '3650', '-sha256',
    '-subj', '/CN=webllm Local CA/O=webllm demo',
    '-addext', 'basicConstraints=critical,CA:true',
    '-addext', 'keyUsage=critical,keyCertSign,cRLSign',
  ], dir );

  // 2) 叶证书（825 天是 Chrome 对非公有信任链的上限；这里给 825 更保险）
  const cn = clean[ 0 ] ?? 'localhost';
  run( openssl, [
    'req', '-newkey', 'rsa:2048', '-nodes',
    '-keyout', 'server.key', '-out', 'server.csr', '-sha256',
    '-subj', `/CN=${ cn }/O=webllm demo`,
  ], dir );

  const san = [ 'DNS:localhost', 'IP:127.0.0.1', ...clean.map( ( ip ) => `IP:${ ip }` ) ].join( ',' );
  fs.writeFileSync( path.join( dir, 'ext.cnf' ), [
    'basicConstraints=CA:FALSE',
    'keyUsage=digitalSignature,keyEncipherment',
    'extendedKeyUsage=serverAuth',
    `subjectAltName=${ san }`,
    '',
  ].join( '\n' ) );

  run( openssl, [
    'x509', '-req', '-in', 'server.csr',
    '-CA', 'ca.pem', '-CAkey', 'ca.key', '-CAcreateserial',
    '-out', 'server.pem', '-days', '825', '-sha256',
    '-extfile', 'ext.cnf',
  ], dir );

  fs.writeFileSync( stampFile, stamp );
  return { dir, ca: caPem, cert: srvPem, key: srvKey, openssl, regenerated: true, ips: clean };
}

/** 默认存放位置。 */
export function defaultCertDir ( root )
{
  return path.join( root, 'node_modules', '.cache', 'certs' );
}

export const tmpDirHint = os.tmpdir;
