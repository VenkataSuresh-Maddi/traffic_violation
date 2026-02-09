// Function to switch between tabs
function openTab(evt, tabName) {
    const tabcontent = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
        tabcontent[i].classList.remove("active");
    }
    const tablinks = document.getElementsByClassName("tab-link");
    for (let i = 0; i < tablinks.length; i++) {
        tablinks[i].classList.remove("active");
    }
    document.getElementById(tabName).style.display = "block";
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

document.addEventListener('DOMContentLoaded', () => {
    // Set initial tab
    document.querySelector('.tab-link').click();
    
    // Set up functionality for each tab
    handleImageForm();
    handleVideoForm();
    handleWebcamButtons();
    
    // Set up slider value displays
    setupConfidenceSliders();
});

// Update confidence value display when slider is moved
function setupConfidenceSliders() {
    const sliders = document.querySelectorAll('.conf-slider');
    sliders.forEach(slider => {
        const valueSpan = document.getElementById(`${slider.id}-value`);
        if (valueSpan) {
            valueSpan.textContent = slider.value;
            slider.addEventListener('input', () => {
                valueSpan.textContent = slider.value;
            });
        }
    });
}

// Handle Image Detection Form Submission
function handleImageForm() {
    const form = document.getElementById('image-form');
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(form);
        const resultsDiv = document.getElementById('image-results');
        
        resultsDiv.innerHTML = '<p>Processing image...</p>';

        fetch('/detect_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            } else {
                resultsDiv.innerHTML = `
                    <div class="result-container">
                        <div class="result-box">
                            <h3>Original Image</h3>
                            <img src="${data.original_path}" alt="Original Image">
                        </div>
                        <div class="result-box">
                            <h3>Detection Result</h3>
                            <img src="${data.processed_path}" alt="Processed Image">
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            resultsDiv.innerHTML = `<p style="color: red;">An unexpected error occurred.</p>`;
            console.error('Error:', error);
        });
    });
}

// Handle Video Detection Form Submission
function handleVideoForm() {
    const form = document.getElementById('video-form');
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(form);
        const resultsDiv = document.getElementById('video-results');
        
        resultsDiv.innerHTML = '<p>Uploading and processing video... This may take a moment.</p>';

        fetch('/detect_video', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            } else {
                resultsDiv.innerHTML = `
                    <h3>Processed Video</h3>
                    <video controls autoplay loop>
                        <source src="${data.video_path}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                `;
            }
        })
        .catch(error => {
            resultsDiv.innerHTML = `<p style="color: red;">An unexpected error occurred.</p>`;
            console.error('Error:', error);
        });
    });
}

// Handle Webcam Feed Buttons
function handleWebcamButtons() {
    const startBtn = document.getElementById('start-webcam');
    const stopBtn = document.getElementById('stop-webcam');
    const webcamContainer = document.getElementById('webcam-feed-container');
    const webcamFeed = document.getElementById('webcam-feed');
    const confidenceSlider = document.getElementById('webcam-conf');

    startBtn.addEventListener('click', () => {
        const confidence = confidenceSlider.value;
        webcamContainer.style.display = 'block';
        // Add a timestamp and confidence to the URL to prevent caching and pass parameter
        webcamFeed.src = `/webcam_feed?t=${new Date().getTime()}&confidence=${confidence}`;
        startBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
        confidenceSlider.disabled = true; // Disable slider during feed
    });

    stopBtn.addEventListener('click', () => {
        webcamFeed.src = '';
        webcamContainer.style.display = 'none';
        stopBtn.style.display = 'none';
        startBtn.style.display = 'inline-block';
        confidenceSlider.disabled = false; // Re-enable slider
        
        fetch('/stop_webcam', { method: 'POST' });
    });
}