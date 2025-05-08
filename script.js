// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// // Form submission
// document.getElementById('contact-form').addEventListener('submit', (e) => {
//     e.preventDefault();
//     alert('Message sent successfully! I will respond shortly.');
//     e.target.reset();
// });

// Scroll animation for project cards
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = 1;
            entry.target.style.transform = 'translateY(0)';
        }
    });
});

document.querySelectorAll('.project-card').forEach((card) => {
    card.style.opacity = 0;
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'all 0.6s ease-out';
    observer.observe(card);
});
// Object Detection Code
let net;
let isDetecting = false;
let frameCount = 0;
let lastUpdate = Date.now();

async function loadObjectDetection() {
    try {
        net = await cocoSsd.load();
        console.log('AI Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Error loading AI model. Please try refreshing the page.');
    }
}

function updateFPS() {
    const now = Date.now();
    const fps = (frameCount * 1000) / (now - lastUpdate);
    document.getElementById('fpsCounter').textContent = `${fps.toFixed(1)} FPS`;
    frameCount = 0;
    lastUpdate = now;
}

async function detectFrame(video, canvas) {
    if (!isDetecting) return;
    
    try {
        frameCount++;
        const predictions = await net.detect(video);
        const ctx = canvas.getContext('2d');
        const minConfidence = parseFloat(document.getElementById('confidenceSlider').value);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
        ctx.restore();

        predictions.forEach(prediction => {
            if (prediction.score < minConfidence) return;
            
            const allowedClasses = Array.from(document.querySelectorAll('.class-filter:checked'))
                .map(checkbox => checkbox.value);
            if (!allowedClasses.includes(prediction.class)) return;

            const mirroredX = canvas.width - prediction.bbox[0] - prediction.bbox[2];
            const y = prediction.bbox[1];
            const width = prediction.bbox[2];
            const height = prediction.bbox[3];

            // Draw bounding box
            ctx.strokeStyle = '#39FF14';
            ctx.lineWidth = 2;
            ctx.strokeRect(mirroredX, y, width, height);
            
            // Draw label
            ctx.fillStyle = '#1A1A1A';
            ctx.fillRect(mirroredX, y - 20, ctx.measureText(prediction.class).width + 10, 20);
            ctx.fillStyle = '#39FF14';
            ctx.font = '16px monospace';
            ctx.fillText(
                `${prediction.class} ${(prediction.score * 100).toFixed(1)}%`,
                mirroredX + 5,
                y - 5
            );
        });

        if (Date.now() - lastUpdate > 1000) updateFPS();
        requestAnimationFrame(() => detectFrame(video, canvas));
    } catch (error) {
        console.error('Detection error:', error);
        stopDetection();
    }
}

async function startDetection() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment" } 
      });
      const video = document.getElementById('webcam');
      const canvas = document.getElementById('output');
      
      video.srcObject = stream;
      
      video.onloadedmetadata = () => {
        video.play();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        isDetecting = true;
        detectFrame(video, canvas);
        document.getElementById('startButton').style.display = 'none';
        document.getElementById('stopButton').style.display = 'inline-block';
      };
    } catch (error) {
      console.error('Webcam error:', error);
      alert('Error accessing webcam. Please enable camera permissions.');
    }
  }
  
  function stopDetection() {
    isDetecting = false;
    const stream = document.getElementById('webcam').srcObject;
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    document.getElementById('startButton').style.display = 'inline-block';
    document.getElementById('stopButton').style.display = 'none';
  }

document.addEventListener('DOMContentLoaded', () => {
    loadObjectDetection();
    
    document.getElementById('startButton').addEventListener('click', startDetection);
    document.getElementById('stopButton').addEventListener('click', stopDetection);
    
    document.getElementById('confidenceSlider').addEventListener('input', (e) => {
        document.getElementById('confidenceValue').textContent = 
            `${Math.round(e.target.value * 100)}%`;
    });
});
// Chatbot functionality
const chatContainer = document.getElementById('chatContainer');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
const statusText = document.getElementById('statusText');

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message
    addMessage(message, 'user');
    chatInput.value = '';
    
    try {
        statusText.textContent = 'Status: Thinking...';
        showLoadingDots();
        
        // My chatbot endpoint
        const response = await fetch('https://chatbackend-kosi.onrender.com/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        addMessage(data.response, 'bot');
    } catch (error) {
        addMessage("⚠️ Sorry, I'm having trouble connecting. Try again later!", 'bot');
    } finally {
        statusText.textContent = 'Status: Ready';
        hideLoadingDots();
    }
}

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message`;
    messageDiv.textContent = text;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Loading animation
function showLoadingDots() {
    const loading = document.createElement('span');
    loading.id = 'loading';
    loading.className = 'loading-dots';
    loading.textContent = '...';
    chatContainer.appendChild(loading);
}

function hideLoadingDots() {
    const loading = document.getElementById('loading');
    if (loading) loading.remove();
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});