/**
 * 公共训练网络的传输层：WebRTC DataChannel 网状连接 + 极简信令。
 *
 * 拓扑：星形。房间创建者是「汇聚节点」（host / aggregator），其余为普通节点。
 * 信令只负责「牵线」——交换 SDP 与 ICE candidate；一旦 P2P 通道建立，
 * 权重与训练日志全部直连互通，不经过任何服务器。
 *
 * 为什么是 WebRTC 而不是 WebSocket 中继：
 *   - 训练数据与权重不落任何中心服务器，契合「公共产品 / 数据不出本地」
 *   - 同一 WiFi 下走 host candidate，局域网内直连，带宽与延迟最好
 *   - 信令服务器是无状态牵线人，可替换、可自建、可共治
 *
 * 控制帧 = string（JSON），权重帧 = ArrayBuffer。RTCDataChannel 保证有序可靠。
 */
import type { DevCap } from './protocol';

export interface PeerInfo
{
  peerId: string;
  name: string;
  role: 'host' | 'peer';
  device: DevCap;
  joinedAt: number;
}

export type Role = 'host' | 'peer';

/**
 * 传输层统一接口。
 *
 * 有两种实现：
 *   - RoomTransport：WebRTC，跨设备（PC ↔ 手机），需要信令服务器牵线
 *   - LocalBus：BroadcastChannel，同一浏览器的多个标签页，**零服务器**
 *
 * 上层（FedNode / UI）只依赖这个接口，因此换传输不需要改训练逻辑。
 */
export interface Transport
{
  readonly selfId: string;
  readonly isHost: boolean;
  /** 房间内其它节点（不含自己） */
  readonly peerInfos: PeerInfo[];
  /** 当前通道已就绪的节点 id */
  readonly openPeerIds: string[];
  /** 建立底层通道：WebRTC 走信令握手，LocalBus 打开 BroadcastChannel */
  connect (): Promise<void>;
  send ( peerId: string, data: string | ArrayBuffer ): boolean;
  sendControl ( peerId: string, msg: object ): boolean;
  /** 广播，except 用于排除发送者自身；返回送达数 */
  broadcast ( data: string | ArrayBuffer, except?: string ): number;
  /**
   * 向信令服务器上报房间的「自我介绍」，供其它设备在「挑房间」列表里看到。
   * 可选：只有走信令的 RoomTransport 支持；本机总线（LocalBus）没有服务器，用不上。
   */
  announce? ( room: Record<string, unknown> ): void;
  close (): void;
}

interface SignalFrame
{
  t: string;
  [ k: string ]: unknown;
}

/**
 * 单条 P2P 连接。
 *
 * 二进制帧必须分块：RTCDataChannel 有单条消息上限（Chrome 常见 256 KiB，
 * 也可能只有 64 KiB），而 WebGPU tiny-GPT 的权重帧约 425 KB —— 直接发会抛
 * 「Trying to send message larger than max-message-size」。
 * MLP 模型只有 54 KB 所以一直没暴露这个问题。
 *
 * 分块信封：[u32 传输号][u32 序号][u32 总片数][载荷]。
 * 数据通道是有序可靠的，所以按序收齐即拼回，不需要超时重传。
 * 所有二进制帧都走信封（包括小帧），这样接收端不需要猜。
 */
const CHUNK_SIZE = 16 * 1024;
const CHUNK_HEADER = 12;

interface Inbound
{
  total: number;
  parts: ArrayBuffer[];
  got: number;
}

class RoomLink
{
  readonly peerId: string;
  private readonly pc: RTCPeerConnection;
  private dc: RTCDataChannel | null = null;
  private readonly inbox = new Map<number, Inbound>();
  private nextTransferId = 1;

  onMessage: ( data: string | ArrayBuffer ) => void = () => {};
  onOpen: () => void = () => {};
  onClose: () => void = () => {};

  constructor ( peerId: string, pc: RTCPeerConnection, dc: RTCDataChannel | null )
  {
    this.peerId = peerId;
    this.pc = pc;
    if ( dc ) this.attach( dc );
  }

  attach ( dc: RTCDataChannel ): void
  {
    this.dc = dc;
    dc.binaryType = 'arraybuffer';
    dc.onopen = () => this.onOpen();
    dc.onclose = () => this.onClose();
    dc.onmessage = ( ev ) => this.onFrame( ev.data as string | ArrayBuffer );
  }

  private onFrame ( data: string | ArrayBuffer ): void
  {
    if ( typeof data === 'string' )
    {
      this.onMessage( data );
      return;
    }
    const buf = data;
    if ( buf.byteLength < CHUNK_HEADER )
    {
      // 不该出现：所有二进制帧都带信封。当作坏帧丢掉，避免污染解析。
      return;
    }
    const dv = new DataView( buf );
    const id = dv.getUint32( 0 );
    const seq = dv.getUint32( 4 );
    const total = dv.getUint32( 8 );
    const payload = buf.slice( CHUNK_HEADER );

    let entry = this.inbox.get( id );
    if ( !entry )
    {
      entry = { total, parts: new Array<ArrayBuffer>( total ), got: 0 };
      this.inbox.set( id, entry );
    }
    if ( entry.parts[ seq ] === undefined )
    {
      entry.parts[ seq ] = payload;
      entry.got += 1;
    }
    if ( entry.got < entry.total ) return;

    this.inbox.delete( id );
    let size = 0;
    for ( const p of entry.parts ) size += p.byteLength;
    const out = new Uint8Array( size );
    let off = 0;
    for ( const p of entry.parts )
    {
      out.set( new Uint8Array( p ), off );
      off += p.byteLength;
    }
    this.onMessage( out.buffer );
  }

  get pcRef (): RTCPeerConnection
  {
    return this.pc;
  }

  get isOpen (): boolean
  {
    return this.dc?.readyState === 'open';
  }

  send ( data: string | ArrayBuffer ): boolean
  {
    if ( !this.dc || this.dc.readyState !== 'open' ) return false;
    if ( typeof data === 'string' )
    {
      this.dc.send( data );
      return true;
    }

    const buf = data;
    const total = Math.max( 1, Math.ceil( buf.byteLength / CHUNK_SIZE ) );
    const id = this.nextTransferId++;
    const src = new Uint8Array( buf );
    for ( let seq = 0; seq < total; seq++ )
    {
      const start = seq * CHUNK_SIZE;
      const end = Math.min( start + CHUNK_SIZE, buf.byteLength );
      const msg = new ArrayBuffer( CHUNK_HEADER + ( end - start ) );
      const dv = new DataView( msg );
      dv.setUint32( 0, id );
      dv.setUint32( 4, seq );
      dv.setUint32( 8, total );
      new Uint8Array( msg, CHUNK_HEADER ).set( src.subarray( start, end ) );
      this.dc.send( msg );
    }
    if ( this.nextTransferId > 0xfffffff0 ) this.nextTransferId = 1;
    return true;
  }

  close (): void
  {
    try { this.dc?.close(); } catch { /* 忽略 */ }
    try { this.pc.close(); } catch { /* 忽略 */ }
  }
}

/** 信令握手帧（由服务端转发）。 */
interface RelayFrame
{
  t: 'signal';
  from: string;
  data: { sdp?: RTCSessionDescriptionInit; ice?: RTCIceCandidateInit };
}

export interface TransportOptions
{
  signalUrl: string;
  roomId: string;
  self: PeerInfo;
  /** 收到控制帧 */
  onControl: ( peerId: string, msg: SignalFrame ) => void;
  /** 收到权重帧 */
  onBinary: ( peerId: string, buf: ArrayBuffer ) => void;
  /** 对端通道就绪 */
  onPeerOpen: ( peerId: string ) => void;
  /** 对端断开 */
  onPeerClose: ( peerId: string ) => void;
  /** 房间成员变化（含自己） */
  onRoster: ( peers: PeerInfo[] ) => void;
  onStatus: ( text: string ) => void;
}

export class RoomTransport implements Transport
{
  private readonly opts: TransportOptions;
  private ws: WebSocket | null = null;
  private readonly links = new Map<string, RoomLink>();
  private readonly pendingIce = new Map<string, RTCIceCandidateInit[]>();
  private roster: PeerInfo[] = [];
  private closed = false;

  constructor ( opts: TransportOptions )
  {
    this.opts = opts;
  }

  get isHost (): boolean
  {
    return this.opts.self.role === 'host';
  }

  get peerInfos (): PeerInfo[]
  {
    return this.roster;
  }

  get openPeerIds (): string[]
  {
    return [ ...this.links.values() ].filter( ( l ) => l.isOpen ).map( ( l ) => l.peerId );
  }

  get selfId (): string
  {
    return this.opts.self.peerId;
  }

  // ---------------------------------------------------------------- 信令

  connect (): Promise<void>
  {
    return new Promise( ( resolve, reject ) =>
    {
      let ws: WebSocket;
      try
      {
        ws = new WebSocket( this.opts.signalUrl );
      }
      catch ( err )
      {
        reject( new Error( `信令地址无效：${ ( err as Error ).message }` ) );
        return;
      }
      this.ws = ws;

      const timer = window.setTimeout( () =>
      {
        reject( new Error( `连接信令服务器超时：${ this.opts.signalUrl }` ) );
      }, 12000 );

      ws.onopen = () =>
      {
        window.clearTimeout( timer );
        this.opts.onStatus( '已连接信令服务器，等待牵线…' );
        this.post( { t: 'join', roomId: this.opts.roomId, peer: this.opts.self } );
        resolve();
      };
      ws.onerror = () =>
      {
        window.clearTimeout( timer );
        reject( new Error( `信令连接失败：${ this.opts.signalUrl }（信令服务是否已启动？）` ) );
      };
      ws.onclose = () =>
      {
        if ( !this.closed ) this.opts.onStatus( '信令连接已断开' );
      };
      ws.onmessage = ( ev ) => this.onSignal( JSON.parse( String( ev.data ) ) as SignalFrame );
    } );
  }

  private post ( obj: unknown ): void
  {
    if ( this.ws?.readyState === WebSocket.OPEN ) this.ws.send( JSON.stringify( obj ) );
  }

  private onSignal ( frame: SignalFrame ): void
  {
    switch ( frame.t )
    {
      case 'joined':
      {
        this.roster = ( frame.peers as PeerInfo[] ) ?? [];
        this.opts.onRoster( this.roster );
        // 若我在他人之后进入房间，且我是 host，需要主动向在场节点发起连接
        if ( this.isHost ) for ( const p of this.roster ) this.offerTo( p.peerId );
        break;
      }
      case 'peer-joined':
      {
        const p = frame.peer as PeerInfo;
        this.roster = [ ...this.roster.filter( ( x ) => x.peerId !== p.peerId ), p ];
        this.opts.onRoster( this.roster );
        this.opts.onStatus( `${ p.name } 加入房间` );
        if ( this.isHost ) this.offerTo( p.peerId );
        break;
      }
      case 'peer-left':
      {
        const id = String( frame.peerId );
        this.roster = this.roster.filter( ( x ) => x.peerId !== id );
        this.opts.onRoster( this.roster );
        this.dropLink( id );
        break;
      }
      case 'signal':
      {
        const relay = frame as unknown as RelayFrame;
        void this.handleRelay( relay.from, relay.data );
        break;
      }
      case 'error':
        this.opts.onStatus( `信令错误：${ String( frame.message ) }` );
        break;
      default:
        break;
    }
  }

  private async handleRelay ( from: string, data: RelayFrame[ 'data' ] ): Promise<void>
  {
    if ( data.sdp )
    {
      const link = this.ensureLink( from, false );
      const pc = link.pcRef;
      await pc.setRemoteDescription( data.sdp );
      // 补投在 setRemoteDescription 之前到达的 candidate
      const queued = this.pendingIce.get( from ) ?? [];
      this.pendingIce.delete( from );
      for ( const c of queued )
      {
        try { await pc.addIceCandidate( c ); } catch { /* 忽略过期 candidate */ }
      }
      if ( data.sdp.type === 'offer' )
      {
        const answer = await pc.createAnswer();
        await pc.setLocalDescription( answer );
        this.post( { t: 'signal', to: from, data: { sdp: pc.localDescription } } );
      }
      return;
    }
    if ( data.ice )
    {
      const link = this.links.get( from );
      if ( !link || !link.pcRef.remoteDescription )
      {
        const q = this.pendingIce.get( from ) ?? [];
        q.push( data.ice );
        this.pendingIce.set( from, q );
        return;
      }
      try { await link.pcRef.addIceCandidate( data.ice ); } catch { /* 忽略 */ }
    }
  }

  private ensureLink ( peerId: string, initiator: boolean ): RoomLink
  {
    const existing = this.links.get( peerId );
    if ( existing ) return existing;

    const pc = new RTCPeerConnection( {
      iceServers: [ { urls: [ 'stun:stun.l.google.com:19302', 'stun:stun1.l.google.com:19302' ] } ],
    } );
    const link = new RoomLink( peerId, pc, null );
    this.links.set( peerId, link );

    pc.onicecandidate = ( ev ) =>
    {
      if ( ev.candidate ) this.post( { t: 'signal', to: peerId, data: { ice: ev.candidate.toJSON() } } );
    };
    pc.onconnectionstatechange = () =>
    {
      const st = pc.connectionState;
      if ( st === 'failed' || st === 'closed' || st === 'disconnected' )
      {
        this.opts.onStatus( `与 ${ peerId.slice( 0, 6 ) } 的连接 ${ st }` );
        if ( st !== 'disconnected' ) this.dropLink( peerId );
      }
    };
    if ( initiator ) link.attach( pc.createDataChannel( 'fed', { ordered: true } ) );
    else pc.ondatachannel = ( ev ) => link.attach( ev.channel );

    link.onOpen = () =>
    {
      this.opts.onPeerOpen( peerId );
    };
    link.onClose = () =>
    {
      this.opts.onPeerClose( peerId );
    };
    link.onMessage = ( data ) =>
    {
      if ( typeof data === 'string' ) this.opts.onControl( peerId, JSON.parse( data ) as SignalFrame );
      else this.opts.onBinary( peerId, data );
    };
    return link;
  }

  private async offerTo ( peerId: string ): Promise<void>
  {
    const link = this.ensureLink( peerId, true );
    const pc = link.pcRef;
    if ( pc.signalingState !== 'stable' ) return;
    try
    {
      const offer = await pc.createOffer();
      await pc.setLocalDescription( offer );
      this.post( { t: 'signal', to: peerId, data: { sdp: pc.localDescription } } );
    }
    catch ( err )
    {
      this.opts.onStatus( `向 ${ peerId.slice( 0, 6 ) } 发起连接失败：${ ( err as Error ).message }` );
    }
  }

  private dropLink ( peerId: string ): void
  {
    const link = this.links.get( peerId );
    if ( !link ) return;
    this.links.delete( peerId );
    link.close();
    this.opts.onPeerClose( peerId );
  }

  // ---------------------------------------------------------------- 发送

  send ( peerId: string, data: string | ArrayBuffer ): boolean
  {
    return this.links.get( peerId )?.send( data ) ?? false;
  }

  sendControl ( peerId: string, msg: object ): boolean
  {
    return this.send( peerId, JSON.stringify( msg ) );
  }

  /** 房间自我介绍：信令服务器只做只读展示，供其它设备「挑房间」。 */
  announce ( room: Record<string, unknown> ): void
  {
    this.post( { t: 'announce', room } );
  }

  /** 广播；except 用于排除发送者自身。返回成功送达的连接数。 */
  broadcast ( data: string | ArrayBuffer, except?: string ): number
  {
    let n = 0;
    for ( const [ id, link ] of this.links )
    {
      if ( id === except ) continue;
      if ( link.send( data ) ) n += 1;
    }
    return n;
  }

  close (): void
  {
    this.closed = true;
    this.post( { t: 'leave' } );
    for ( const id of [ ...this.links.keys() ] ) this.dropLink( id );
    try { this.ws?.close(); } catch { /* 忽略 */ }
  }
}
