const cube = document.querySelector('.cube');
let isDragging = false;
let initialX, initialY;

// Handle drag events for the cube
cube.addEventListener('mousedown', startDrag);
cube.addEventListener('mousemove', moveCube);
document.addEventListener('mouseup', stopDrag);

function startDrag(event) {
    isDragging = true;
    initialX = event.clientX - cube.offsetLeft;
    initialY = event.clientY - cube.offsetTop;
}

function moveCube(event) {
    if (!isDragging) return;

    const x = event.clientX - initialX;
    const y = event.clientY - initialY;

    // Update the cube's position based on mouse movement
    cube.style.left = `${x}px`;
    cube.style.top = `${y}px`;

    // Adjust the cube's position to simulate gravity
    adjustGravity();
}

function stopDrag() {
    isDragging = false;
}

// Function to adjust the cube's position due to gravity
function adjustGravity() {
    const windowHeight = window.innerHeight;
    const cubeHeight = 50; // Assuming the cube's height is always 50px for simplicity

    if (cube.offsetTop + cubeHeight >= windowHeight) {
        // If the cube reaches the bottom edge of the window, it falls back down
        cube.style.transform = 'translateY(100% - 50px)';
    } else {
        // Otherwise, maintain its current position without falling further
        cube.style.transform = `translateY(${cube.offsetTop}px)`;
    }
}