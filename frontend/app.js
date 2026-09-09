/**
 * Syllabus Assistant - Frontend Logic (app.js)
 * Wires UI components to the FastAPI RAG backend endpoints (/upload, /ask).
 */

document.addEventListener('DOMContentLoaded', () => {
    // API Base URL
    const API_BASE_URL = 'http://localhost:8000';

    // DOM References
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const selectFileBtn = document.getElementById('selectFileBtn');
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileDetails = document.getElementById('fileDetails');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const fileSizeDisplay = document.getElementById('fileSizeDisplay');
    const uploadStatus = document.getElementById('uploadStatus');

    const chatMessages = document.getElementById('chatMessages');
    const chatForm = document.getElementById('chatForm');
    const questionInput = document.getElementById('questionInput');
    const sendBtn = document.getElementById('sendBtn');
    const suggestedChips = document.querySelectorAll('.chip');
    const statusBadge = document.getElementById('statusBadge');
    const currentDocName = document.getElementById('currentDocName');
    const clearChatBtn = document.getElementById('clearChatBtn');

    // State Variables
    let selectedFile = null;
    let isFileUploaded = false;
    let isWaitingForAnswer = false;

    // Initialize UI State
    updateUIState();

    // --- FILE SELECTION & DRAG-AND-DROP ---

    if (selectFileBtn && fileInput) {
        selectFileBtn.addEventListener('click', (e) => {
            e.preventDefault();
            fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    if (dropZone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('drag-over');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('drag-over');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                handleFileSelection(dt.files[0]);
            }
        });
    }

    function handleFileSelection(file) {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showStatus(uploadStatus, '⚠️ Please select a valid PDF document (.pdf)', 'error');
            return;
        }
        selectedFile = file;
        if (fileNameDisplay) fileNameDisplay.textContent = file.name;
        if (fileSizeDisplay) fileSizeDisplay.textContent = formatBytes(file.size);
        if (fileDetails) fileDetails.classList.remove('hidden');
        if (uploadBtn) uploadBtn.disabled = false;
        showStatus(uploadStatus, '📄 File selected. Click "Index Syllabus PDF" to process.', 'info');
    }

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // --- UPLOAD PROCESSOR ---

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!selectedFile) {
                showStatus(uploadStatus, 'Please select a PDF file first.', 'error');
                return;
            }
            await processFileUpload(selectedFile);
        });
    }

    async function processFileUpload(file) {
        const formData = new FormData();
        formData.append('file', file);

        setUploadLoading(true);
        showStatus(uploadStatus, '⚡ Extracting text & generating vector embeddings...', 'info');

        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || errorData.message || `Upload failed with status code ${response.status}`);
            }

            const data = await response.json();
            isFileUploaded = true;
            setUploadLoading(false);

            const chunksMsg = data.num_chunks ? ` (${data.num_chunks} vector chunks indexed)` : '';
            showStatus(uploadStatus, `✅ Syllabus successfully indexed!${chunksMsg}`, 'success');

            if (statusBadge) {
                statusBadge.textContent = 'Active Syllabus';
                statusBadge.className = 'status-badge active';
            }
            if (currentDocName) {
                currentDocName.textContent = file.name;
            }

            updateUIState();
            appendMessage('system', `📄 **${file.name}** has been uploaded and indexed. You can now ask questions!`);

        } catch (error) {
            console.error('Upload Error:', error);
            setUploadLoading(false);
            showStatus(uploadStatus, `❌ Upload Error: ${error.message || 'Could not connect to backend server. Make sure FastAPI is running on port 8000.'}`, 'error');
        }
    }

    function setUploadLoading(isLoading) {
        if (!uploadBtn) return;
        uploadBtn.disabled = isLoading;
        if (isLoading) {
            uploadBtn.innerHTML = '<span class="spinner"></span> Processing PDF...';
        } else {
            uploadBtn.innerHTML = '🚀 Index Syllabus PDF';
        }
    }

    // --- QUESTION & CHAT HANDLER ---

    if (chatForm) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = questionInput.value.trim();
            if (!question || isWaitingForAnswer) return;

            await submitQuestion(question);
        });
    }

    if (questionInput) {
        questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event('submit'));
            }
        });
    }

    suggestedChips.forEach(chip => {
        chip.addEventListener('click', () => {
            if (!isFileUploaded) {
                showStatus(uploadStatus, '⚠️ Please upload and index a syllabus PDF first.', 'error');
                return;
            }
            const promptText = chip.getAttribute('data-prompt') || chip.textContent.trim();
            if (questionInput) {
                questionInput.value = promptText;
            }
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', () => {
            if (chatMessages) {
                chatMessages.innerHTML = '';
                appendMessage('system', 'Chat reset. Ask any question about your syllabus!');
            }
        });
    }

    async function submitQuestion(question) {
        appendMessage('user', question);
        if (questionInput) questionInput.value = '';

        isWaitingForAnswer = true;
        updateChatInputState();

        const typingId = showTypingIndicator();

        try {
            const response = await fetch(`${API_BASE_URL}/ask`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            removeTypingIndicator(typingId);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || errorData.message || `Server error ${response.status}`);
            }

            const data = await response.json();
            const answer = data.answer || data.response || "I couldn't find that in the syllabus.";
            const sources = data.sources || data.context_chunks || [];

            appendMessage('bot', answer, { sources });

        } catch (error) {
            console.error('Ask Error:', error);
            removeTypingIndicator(typingId);
            appendMessage('bot', `⚠️ **Error:** ${error.message || 'Failed to get answer from backend backend.'}`, { isError: true });
        } finally {
            isWaitingForAnswer = false;
            updateChatInputState();
            if (questionInput) questionInput.focus();
        }
    }

    // --- HELPER FUNCTIONS ---

    function updateUIState() {
        if (questionInput) {
            questionInput.disabled = !isFileUploaded || isWaitingForAnswer;
            questionInput.placeholder = isFileUploaded
                ? "Ask a question about midterm dates, grading breakdown, office hours..."
                : "Please upload and index a syllabus PDF first...";
        }
        if (sendBtn) {
            sendBtn.disabled = !isFileUploaded || isWaitingForAnswer;
        }
        suggestedChips.forEach(chip => {
            if (isFileUploaded) {
                chip.classList.remove('disabled');
            } else {
                chip.classList.add('disabled');
            }
        });
    }

    function updateChatInputState() {
        if (sendBtn) sendBtn.disabled = isWaitingForAnswer;
        if (questionInput) questionInput.disabled = isWaitingForAnswer;
    }

    function appendMessage(sender, text, options = {}) {
        if (!chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message ${options.isError ? 'error-message' : ''}`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        if (sender === 'user') avatar.innerHTML = '👤';
        else if (sender === 'bot') avatar.innerHTML = '🤖';
        else avatar.innerHTML = '💡';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const textWrapper = document.createElement('div');
        textWrapper.className = 'message-text';
        textWrapper.innerHTML = formatMarkdown(text);

        contentDiv.appendChild(textWrapper);

        if (options.sources && Array.isArray(options.sources) && options.sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources-container';
            sourcesDiv.innerHTML = `<span class="sources-title">📌 Relevant Context Chunks:</span>`;
            
            options.sources.forEach((src, idx) => {
                const sourceBadge = document.createElement('div');
                sourceBadge.className = 'source-item';
                const snippet = typeof src === 'string' ? src : (src.text || JSON.stringify(src));
                sourceBadge.innerHTML = `<span class="source-tag">Chunk ${idx + 1}</span> ${escapeHtml(snippet.substring(0, 150))}${snippet.length > 150 ? '...' : ''}`;
                sourcesDiv.appendChild(sourceBadge);
            });
            
            contentDiv.appendChild(sourcesDiv);
        }

        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        contentDiv.appendChild(timeSpan);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);

        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.id = id;
        typingDiv.className = 'message bot-message typing-indicator-msg';
        typingDiv.innerHTML = `
            <div class="avatar">🤖</div>
            <div class="message-content">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        if (chatMessages) {
            chatMessages.appendChild(typingDiv);
            scrollToBottom();
        }
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    function showStatus(element, message, type = 'info') {
        if (!element) return;
        element.textContent = message;
        element.className = `status-banner ${type}`;
        element.classList.remove('hidden');
    }

    function formatMarkdown(text) {
        if (!text) return '';
        let escaped = escapeHtml(text);
        escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        escaped = escaped.replace(/\*(.*?)\*/g, '<em>$1</em>');
        escaped = escaped.replace(/`(.*?)`/g, '<code>$1</code>');
        escaped = escaped.replace(/\n/g, '<br>');
        return escaped;
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
