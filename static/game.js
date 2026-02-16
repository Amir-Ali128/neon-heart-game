let canvas = document.getElementById("game");
let ctx = canvas.getContext("2d");

let player = {x:200,y:550,size:20};
let hearts=[];
let score=0;
let running=false;

function startGame(){
  running=true;
  score=0;
  hearts=[];
  document.getElementById("msg").innerText="Viel Glück Sara 💜";
}

function spawnHeart(){
  hearts.push({
    x:Math.random()*380,
    y:0,
    size:15
  });
}

setInterval(()=>{
  if(running) spawnHeart();
},1000);

function update(){
  if(!running) return;

  ctx.clearRect(0,0,400,600);

  ctx.fillStyle="pink";
  ctx.fillRect(player.x,player.y,player.size,player.size);

  hearts.forEach(h=>{
    h.y+=3;
    ctx.fillRect(h.x,h.y,h.size,h.size);

    if(Math.abs(h.x-player.x)<20 && Math.abs(h.y-player.y)<20){
      score++;
      document.getElementById("msg").innerText =
        "Score: " + score + " | Beste Schwester 💜";
    }
  });

  requestAnimationFrame(update);
}

update();

canvas.addEventListener("mousemove",(e)=>{
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (400 / rect.width);
  player.x = Math.max(0, Math.min(380, x));
});

canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches[0];
  const x = (touch.clientX - rect.left) * (400 / rect.width);
  player.x = Math.max(0, Math.min(380, x));
}, { passive: false });
