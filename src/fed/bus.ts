/**
 * 本机多标签页传输：基于 BroadcastChannel，**不需要任何服务器**。
 *
 * 用途有二：
 *   1. 产品上：同一台机器开两个标签页就能立刻看到联邦训练跑起来 ——
 *      零配置、离线可用，是「开箱即可」这个目标的最短路径。
 *   2. 工程上：它和 WebRTC 走的是**同一个 Transport 接口**，
 *      于是可以在 Node 里用两个 FedNode 实例跑完整的端到端测试
 *      （见 scripts/verify-e2e.mjs），把「开轮 → 上报 → 校验 → 聚合 → 广播」
 *      这条最容易出错的编排逻辑真正验证一遍，而不是只测零件。
 *
 * 与 WebRTC 的语义对齐：控制帧走 JSON 字符串，权重帧走 ArrayBuffer；
 * 通过 structured clone 直接传二进制，无需序列化。
 */
import type { PeerInfo } from './transport';
import type { Transport } from './transport';

export interface LocalBusOptions
{
  roomId: string;
  self: PeerInfo;
  onControl: ( peerId: string, msg: unknown ) => void;
  onBinary: ( peerId: string, buf: ArrayBuffer ) => void;
  onPeerOpen: ( peerId: string ) => void;
  onPeerClose: ( peerId: string ) => void;
  onRoster: ( peers: PeerInfo[] ) => void;
  onStatus: ( text: string ) => void;
}

type BusFrame =
  | { kind: 'present'; from: string; peer: PeerInfo }
  | { kind: 'bye'; from: string }
  | { kind: 'data'; from: string; to: string | null; payload: string | ArrayBuffer };

export class LocalBus implements Transport
{
  private readonly o: LocalBusOptions;
  private ch: BroadcastChannel | null = null;
  private readonly peers = new Map<string, PeerInfo>();
  private readonly open = new Set<string>();
  private closed = false;
  /** 已经握过手的节点，避免 present 互相触发无限回环 */
  private readonly greeted = new Set<string>();

  constructor ( opts: LocalBusOptions )
  {
    this.o = opts;
  }

  get selfId (): string { return this.o.self.peerId; }
  get isHost (): boolean { return this.o.self.role === 'host'; }
  get peerInfos (): PeerInfo[] { return [ ...this.peers.values() ]; }
  get openPeerIds (): string[] { return [ ...this.open ]; }

  connect (): Promise<void>
  {
    return new Promise( ( resolve, reject ) =>
    {
      try
      {
        this.ch = new BroadcastChannel( `wb-fed-${ this.o.roomId }` );
      }
      catch ( err )
      {
        reject( new Error( `BroadcastChannel 不可用：${ ( err as Error ).message }` ) );
        return;
      }
      this.ch.onmessage = ( ev ) => this.onFrame( ev.data as BusFrame );
      this.o.onStatus( `已加入本机总线（房间 ${ this.o.roomId }），等待其他标签页…` );
      this.post( { kind: 'present', from: this.selfId, peer: this.o.self } );
      resolve();
    } );
  }

  private post ( frame: BusFrame ): void
  {
    if ( this.closed || !this.ch ) return;
    this.ch.postMessage( frame );
  }

  private onFrame ( frame: BusFrame ): void
  {
    if ( this.closed || !frame || frame.from === this.selfId ) return;

    if ( frame.kind === 'present' )
    {
      const isNew = !this.peers.has( frame.from );
      this.peers.set( frame.from, frame.peer );
      if ( isNew )
      {
        // 回通告一次，让新来的标签页也能看到我（下一轮起 greeted 会挡住重复）
        if ( !this.greeted.has( frame.from ) )
        {
          this.greeted.add( frame.from );
          this.post( { kind: 'present', from: this.selfId, peer: this.o.self } );
        }
        this.markOpen( frame.from );
        this.o.onRoster( this.peerInfos );
      }
      return;
    }

    if ( frame.kind === 'bye' )
    {
      this.peers.delete( frame.from );
      this.open.delete( frame.from );
      this.greeted.delete( frame.from );
      this.o.onPeerClose( frame.from );
      this.o.onRoster( this.peerInfos );
      return;
    }

    if ( frame.to && frame.to !== this.selfId ) return;
    if ( typeof frame.payload === 'string' ) this.o.onControl( frame.from, JSON.parse( frame.payload ) as unknown );
    else this.o.onBinary( frame.from, frame.payload );
  }

  private markOpen ( peerId: string ): void
  {
    if ( this.open.has( peerId ) ) return;
    this.open.add( peerId );
    this.o.onPeerOpen( peerId );
  }

  send ( peerId: string, data: string | ArrayBuffer ): boolean
  {
    if ( !this.peers.has( peerId ) ) return false;
    this.post( { kind: 'data', from: this.selfId, to: peerId, payload: data } );
    return true;
  }

  sendControl ( peerId: string, msg: object ): boolean
  {
    return this.send( peerId, JSON.stringify( msg ) );
  }

  broadcast ( data: string | ArrayBuffer, except?: string ): number
  {
    let n = 0;
    for ( const id of this.peers.keys() )
    {
      if ( id === except ) continue;
      if ( this.send( id, data ) ) n += 1;
    }
    return n;
  }

  close (): void
  {
    if ( this.closed ) return;
    this.post( { kind: 'bye', from: this.selfId } );
    this.closed = true;
    try { this.ch?.close(); } catch { /* 忽略 */ }
    this.ch = null;
  }
}
