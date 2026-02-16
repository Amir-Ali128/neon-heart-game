const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

let player = { x: 180, y: 550, size: 20 };
let hearts = [];
let score = 0;
let running = false;

// Buton fonksiyonu
function startGame() {
    running = true;
    hearts = [];
    score = 0;
    document.getElementById("msg").innerText = "Spiel läuft... 💜";
    requestAnimationFrame(gameLoop);
}

// Kalp oluştur
function spawnHeart() {
    hearts.push({
        x: Math.random() * (canvas.width - 20),
        y: 0,
        size: 15
    });
}

// Her 1 saniyede kalp
setInterval(() => {
    if (running) spawnHeart();
}, 1000);

// Oyun döngüsü
function gameLoop() {

    if (!running) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Player çiz
    ctx.fillStyle = "pink";
    ctx.fillRect(player.x, player.y, player.size, player.size);

    // Kalpler
    hearts.forEach((heart, index) => {

        heart.y += 3;

        ctx.fillRect(heart.x, heart.y, heart.size, heart.size);

        if (
            heart.x < player.x + player.size &&
            heart.x + heart.size > player.x &&
            heart.y < player.y + player.size &&
            heart.y + heart.size > player.y
        ) {
            hearts.splice(index, 1);
            score++;
            document.getElementById("msg").innerText =
                "Score: " + score + " 💜";
        }

    });

    requestAnimationFrame(gameLoop);
}

// Mouse kontrol
canvas.addEventListener("mousemove", e => {
    const rect = canvas.getBoundingClientRect();
    player.x = e.clientX - rect.left - player.size / 2;
});

// Touch kontrol
canvas.addEventListener("touchmove", e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    player.x = e.touches[0].clientX - rect.left - player.size / 2;
});
