"""Local browser monitor, served by webcam_stream.py; all inference runs on Pi."""

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Webcam laptop · Pi monitor</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#10151d;color:#e9eef5;font:16px system-ui,sans-serif}
main{max-width:1360px;margin:auto;padding:28px}h1{font-size:26px;margin:0 0 8px}
p{color:#aebdcd;line-height:1.5}section{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:24px}
article{min-width:0;background:#1a2431;border:1px solid #334155;border-radius:12px;overflow:hidden}
header{padding:16px}h2{font-size:18px;margin:0 0 8px}.status{color:#f7c66a;font-size:14px;min-height:22px}
.live{color:#7ce4b5}.stale{color:#ff9a9a}.picture{aspect-ratio:4/3;background:#080b10;display:flex;align-items:center;justify-content:center}
img{width:100%;height:100%;object-fit:contain;display:none}.placeholder{color:#aebdcd;padding:24px;text-align:center}
footer{font-size:14px;color:#aebdcd;margin-top:20px;line-height:1.6}code{color:#d4e5ff}
@media(max-width:800px){section{grid-template-columns:1fr}main{padding:18px}}
</style></head><body><main>
<h1>Webcam laptop → Raspberry Pi</h1>
<p>Ảnh bên phải được Pi xử lý và gửi về. Laptop chỉ gửi camera và hiển thị kết quả.</p>
<section>
<article><header><h2>Camera laptop · ảnh gốc</h2><div id="raw-status" class="status">Đang kết nối camera…</div></header>
<div class="picture"><img id="raw-image" alt="Ảnh webcam gốc"><div id="raw-placeholder" class="placeholder">Chờ frame từ webcam laptop</div></div></article>
<article><header><h2>Kết quả từ Pi · landmark và nhãn</h2><div id="pi-status" class="status">Chưa nhận kết quả từ Pi</div></header>
<div class="picture"><img id="pi-image" alt="Ảnh đã được Pi xử lý"><div id="pi-placeholder" class="placeholder">Chạy run_video.py trên Pi với --return-preview</div></div></article>
</section>
<footer>Pi OS Lite vẫn chạy <code>--headless</code>. Không cần desktop trên Pi.<br>
Nếu kết quả ngừng cập nhật hơn 2 giây, trạng thái chuyển đỏ; ảnh cuối được giữ để đối chiếu.</footer>
</main><script>
const states={raw:{seq:-1,updated:null,age:0,url:null,error:null},pi:{seq:-1,updated:null,age:0,url:null,error:null}};
function showFrame(name,blob){const s=states[name],img=document.getElementById(name+'-image'),old=s.url;
 s.url=URL.createObjectURL(blob);img.onload=()=>{if(old)URL.revokeObjectURL(old)};
 img.src=s.url;img.style.display='block';document.getElementById(name+'-placeholder').style.display='none';}
function refreshStatus(name){const s=states[name],el=document.getElementById(name+'-status');
 if(s.error){el.textContent=s.error;el.className='status stale';return;}
 if(s.updated===null){el.textContent=name==='pi'?'Chưa nhận kết quả từ Pi · cần --return-preview':'Đang chờ camera';el.className='status';return;}
 const age=s.age+(performance.now()-s.updated)/1000,stale=age>2;
 el.textContent=(stale?'Không có frame mới':'Đang cập nhật')+' · frame '+s.seq+' · '+age.toFixed(1)+' giây trước';
 el.className='status '+(stale?'stale':'live');}
async function poll(name){const s=states[name],path=name==='raw'?'/frame.jpg':'/result.jpg';
 try{const response=await fetch(path+'?after='+s.seq,{cache:'no-store'});
  if(response.status!==204){if(!response.ok)throw Error('HTTP '+response.status);
   const seq=Number(response.headers.get('X-Frame-Id'));
   const age=Number(response.headers.get(name==='raw'?'X-Frame-Age-Ms':'X-Preview-Age-Ms'))/1000;
   const blob=await response.blob();showFrame(name,blob);s.seq=seq;s.age=age;s.updated=performance.now();}
  s.error=null;
 }catch(error){s.error=name==='raw'?'Mất kết nối camera / sender':'Mất kết nối trang nhận kết quả';}
 refreshStatus(name);setTimeout(()=>poll(name),s.error?1000:(name==='raw'?30:200));}
setInterval(()=>{refreshStatus('raw');refreshStatus('pi')},250);poll('raw');poll('pi');
</script></body></html>"""
