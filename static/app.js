// Client-side interactions for PhishGuard Cybersecurity Scanner

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // Tab Switching
    // -------------------------------------------------------------
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetElement = document.getElementById(targetTab);
            if (targetElement) {
                targetElement.classList.add('active');
            }
        });
    });

    // -------------------------------------------------------------
    // Quick Sample URL Fill
    // -------------------------------------------------------------
    const samplePills = document.querySelectorAll('.sample-pill');
    const urlInput = document.getElementById('url-input');

    samplePills.forEach(pill => {
        pill.addEventListener('click', () => {
            const sampleUrl = pill.getAttribute('data-url');
            if (urlInput && sampleUrl) {
                urlInput.value = sampleUrl;
                urlInput.focus();
            }
        });
    });

    // -------------------------------------------------------------
    // QR Drag and Drop & File Preview
    // -------------------------------------------------------------
    const dropzone = document.getElementById('qr-dropzone');
    const fileInput = document.getElementById('qr-file-input');
    const filePreview = document.getElementById('file-preview');
    const fileNameSpan = document.getElementById('file-name');

    if (dropzone && fileInput) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                updateFilePreview(fileInput.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files.length > 0) {
                updateFilePreview(fileInput.files[0]);
            }
        });
    }

    function updateFilePreview(file) {
        if (!filePreview || !fileNameSpan || !file) return;
        fileNameSpan.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        filePreview.style.display = 'flex';
    }

    // -------------------------------------------------------------
    // Copy URL to Clipboard
    // -------------------------------------------------------------
    window.copyUrlText = function (url) {
        if (!navigator.clipboard) return;
        navigator.clipboard.writeText(url).then(() => {
            alert('URL copied to clipboard!');
        }).catch(err => {
            console.error('Could not copy text: ', err);
        });
    };
});
