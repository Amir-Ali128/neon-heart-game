let canvas = document.getElementById("game");
let ctx = canvas.getContext("2d");

let player = { x: 180, y: 550, size: 20 };
let hearts = [];
let score = 0;
let running = false;

// Oyunu başlat
function startGame() {
    running = true;
    score = 0;
    hearts = [];
    document.getElementById("msg").innerText = "Viel Glück! 💜";
    update();
}

// Kalp oluştur
function spawnHeart() {
    hearts.push({
        x: Math.random() * (canvas.width - 20),
        y: 0,
        size: 15
    });
}

setInterval(() => {
    if (running) spawnHeart();
}, 1000);

// Oyun döngüsü
function update() {
    if (!running) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Player çiz
    ctx.fillStyle = "pink";
    ctx.fillRect(player.x, player.y, player.size, player.size);

    // Kalpleri çiz ve kontrol et
    hearts.forEach((h, i) => {
        h.y += 3;
        ctx.fillRect(h.x, h.y, h.size, h.size);

        if (
            h.x < player.x + player.size &&
            h.x + h.size > player.x &&
            h.y < player.y + player.size &&
            h.y + h.size > player.y
        ) {
            hearts.splice(i, 1);
            score++;
            document.getElementById("msg").innerText =
                "Score: " + score + " 💜";
        }
    });

    requestAnimationFrame(update);
}

// Mouse kontrol
canvas.addEventListener("mousemove", function(e) {
    const rect = canvas.getBoundingClientRect();
    player.x = e.clientX - rect.left - player.size / 2;
});

// Touch kontrol
canvas.addEventListener("touchmove", function(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    player.x = e.touches[0].clientX - rect.left - player.size / 2;
});
