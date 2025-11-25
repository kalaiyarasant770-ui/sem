// index.js

// ✅ Get vertical_id from login session (not dropdown)
let SELECTED_VERTICAL_ID = null;

document.addEventListener("DOMContentLoaded", () => {
  if (localStorage.getItem("loggedIn") !== "true") {
    window.location.href = "login.html";
    return;
  }

  // ✅ Load vertical_id from login session
  SELECTED_VERTICAL_ID = parseInt(localStorage.getItem("vertical_id"));
  const verticalName = localStorage.getItem("vertical_name");
  const username = localStorage.getItem("username");

  if (!SELECTED_VERTICAL_ID) {
    alert("Session expired. Please login again.");
    window.location.href = "login.html";
    return;
  }

  console.log(`✅ Session loaded: ${verticalName} (ID: ${SELECTED_VERTICAL_ID})`);
  
  // ✅ Display logged-in vertical name (optional)
  const verticalDisplay = document.getElementById("vertical-display");
  if (verticalDisplay) {
    verticalDisplay.textContent = `${verticalName}`;
  }

  // ... rest of your code (btn, container, messagesDiv, etc.)
  const btn = document.getElementById("chatbot-btn");
  const container = document.getElementById("chatbot-container");
  const messagesDiv = document.getElementById("chatbot-messages");
  const input = document.getElementById("messageInput");
  const sendBtn = document.getElementById("send-btn");
  const uploadStatus = document.getElementById("uploadStatus");
  const fileInput = document.getElementById("fileInput");
  const sessionInfo = document.getElementById("sessionInfo");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  const logoutBtn = document.getElementById("logout-btn");
  const closeBtn = document.getElementById("chatbot-close");

  let isProcessing = false;
  let sessionId = localStorage.getItem("chatbot_session_id");
  let messageCount = 0;

  function initSession() {
    if (!sessionId) {
      sessionId = null;
      sessionInfo.innerHTML = '<i class="fa-solid fa-comment"></i> New conversation';
    } else {
      sessionInfo.innerHTML = `<i class="fa-solid fa-comments"></i> Session active (${messageCount} messages)`;
    }
  }

  function updateSessionInfo() {
    if (sessionId) {
      sessionInfo.innerHTML = `<i class="fa-solid fa-comments"></i> Session active (${messageCount} messages)`;
    }
  }

  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", async () => {
      if (!sessionId) {
        messagesDiv.innerHTML = '';
        messageCount = 0;
        //addMessage("Conversation cleared! Start fresh.", false);
        return;
      }

      //if (!confirm("Clear conversation history? This will start a new session.")) return;

      try {
        const res = await fetch(`${ACTIVE_CONFIG.BASE_URL}/chat/clear`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });

        if (res.ok) {
          messagesDiv.innerHTML = '';
          sessionId = null;
          messageCount = 0;
          localStorage.removeItem("chatbot_session_id");
          sessionInfo.innerHTML = '<i class="fa-solid fa-comment"></i> New conversation';
          //addMessage("Conversation cleared! How can I help you?", false);
        } else {
          // addMessage("Failed to clear conversation. Please try again.", false);
        }
      } catch (err) {
        console.error("Failed to clear:", err);
        //addMessage("Failed to clear conversation. Please try again.", false);
      }
    });
  }

  if (btn && container) {
    btn.addEventListener("click", () => {
      container.style.display = "flex";
      btn.style.display = "none";
      input.focus();
      initSession();
    });
  }

  if (closeBtn && btn && container) {
    closeBtn.addEventListener("click", () => {
      container.style.display = "none";
      btn.style.display = "flex";
    });
  }

  function fmtTime() {
    return new Date().toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
    });
  }

  function copyToClipboard(text, icon) {
    // Extract plain text from HTML if needed
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = text;
    const plainText = tempDiv.textContent || tempDiv.innerText || text;
    
    navigator.clipboard.writeText(plainText).then(() => {
      icon.classList.remove("fa-copy");
      icon.classList.add("fa-check");
      
      setTimeout(() => {
        icon.classList.remove("fa-check");
        icon.classList.add("fa-copy");
      }, 1000);
    }).catch(err => {
      console.error("Failed to copy:", err);
      alert("Copy failed!");
    });
  }


function formatMetricsResponse(content) {
  // Strip the markers
  let formatted = content
    .replace(/\[METRICS_START\]/g, '')
    .replace(/\[METRICS_END\]/g, '')
    .trim();

  const lines = formatted.split('\n');
  let html = '';

  // -----------------------------------------------------------------
  // Helper: safely parse any ISO timestamp and return a readable string
  // -----------------------------------------------------------------
  function safeParseDate(dateStr) {
    if (!dateStr || dateStr === 'N/A') return 'N/A';
    let cleaned = dateStr.trim();

    // If there is no timezone info, assume IST (+05:30)
    if (!/[Z+-]/.test(cleaned) && cleaned.includes('T')) {
      cleaned += '+05:30';
    }

    const d = new Date(cleaned);
    if (isNaN(d.getTime())) return dateStr; // fallback to raw value

    // Indian locale, 24-hour, no comma
    return d.toLocaleString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
      timeZoneName: 'short',
    }).replace(',', '');
  }

  // -----------------------------------------------------------------
  // Process each line
  // -----------------------------------------------------------------
  for (let line of lines) {
    line = line.trim();
    if (!line) continue;

    // Title (Performance header)
    if (line.endsWith('Performance:')) {
      html += `<div class="metric-title">${line}</div>`;
      continue;
    }

    // ----- Time fields ------------------------------------------------
    if (line.startsWith('Start Time:') || line.startsWith('End Time:')) {
      const [label, value] = line.split(/:\s*/, 2);
      const display = safeParseDate(value.trim());
      html += `<div class="metric-line time-info">
                 <span class="metric-label">${label}:</span>
                 <span class="metric-value">${display}</span>
               </div>`;
      continue;
    }

    // ----- Generic key:value -----------------------------------------
    // if (line.includes(':')) {
    //   const colonIdx = line.indexOf(':');
    //   const label = line.substring(0, colonIdx).trim();
    //   const value = line.substring(colonIdx + 1).trim();
    //   html += `<div class="metric-line">
    //              <span class="metric-label">${label}:</span>
    //              <span class="metric-value">${value}</span>
    //            </div>`;
    //   continue;
    // }

        // ----- Note field (special styling) ------------------------------
    if (line.startsWith('Note:')) {
      const [label, value] = line.split(/:\s*/, 2);
      html += `<div class="metric-line-note-line" style="margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);">
                 <span class="metric-label">${label}:</span>
                 <span class="metric-value" style="opacity: 0.9;">${value}</span>
               </div>`;
      continue;
    }

    // ----- Generic key:value -----------------------------------------
    if (line.includes(':')) {
      const colonIdx = line.indexOf(':');
      const label = line.substring(0, colonIdx).trim();
      const value = line.substring(colonIdx + 1).trim();
      html += `<div class="metric-line">
                 <span class="metric-label">${label}:</span>
                 <span class="metric-value">${value}</span>
               </div>`;
      continue;
    }

    // ----- Fallback plain text ---------------------------------------
    html += `<div>${line}</div>`;
  }

  return html;
}

  /**
   * Format advisory response with numbered list
   */
  function formatAdvisoryResponse(content) {
    // Remove markers
    let formatted = content.replace(/\[ADVISORY_START\]/g, '').replace(/\[ADVISORY_END\]/g, '').trim();
    
    // Split into lines and filter empty
    const lines = formatted.split('\n').filter(l => l.trim());
    
    let html = '<ol>';
    for (let line of lines) {
      line = line.trim();
      // Remove existing number if present (1., 2., etc.)
      line = line.replace(/^\d+\.\s*/, '');
      if (line) {
        html += `<li>${line}</li>`;
      }
    }
    html += '</ol>';
    
    return html;
  }

  /**
   * Add message to chat with proper formatting
   */

  function formatResponseTime(seconds) {
  if (seconds < 60) {
    return `${seconds.toFixed(2)}s`;
  }

  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(0);

  return `${mins}m ${secs}s`;
}


  function addMessage(content, isUser = false, metadata = {}) {
    
    const row = document.createElement("div");
    row.className = `msg-row ${isUser ? "user" : "bot"}`;

    const bubble = document.createElement("div");
    bubble.className = `message ${isUser ? "user" : "bot"}`;
    
    // Format bot messages based on type
    if (!isUser) {
      // Check for metrics response
      if (content.includes('[METRICS_START]')) {
        bubble.classList.add('metrics-response');
        bubble.innerHTML = formatMetricsResponse(content);
      }
      // Check for advisory response
      else if (content.includes('[ADVISORY_START]')) {
        bubble.classList.add('advisory-response');
        bubble.innerHTML = formatAdvisoryResponse(content);
      }
      // Regular message - preserve line breaks
      else {
        bubble.innerHTML = content.replace(/\n/g, '<br>');
      }
    } else {
      // User messages - plain text
      bubble.textContent = content;
    }

    const time = document.createElement("div");
    time.className = "msg-time";
    time.textContent = fmtTime();

    const copyIcon = document.createElement("i");
    copyIcon.className = "fa-solid fa-copy copy-btn";
    copyIcon.title = "Copy message";
    copyIcon.onclick = () => copyToClipboard(bubble.innerHTML, copyIcon);
    
    const meta = document.createElement("div");
   meta.innerHTML = metadata.response_time
  ? `<span>⚡ ${formatResponseTime(metadata.response_time)}</span>`
  : "";
      

    const metaRow = document.createElement("div");
    metaRow.className = "meta-row";

    if (!isUser) {
      metaRow.appendChild(time);
      if (meta.innerHTML) metaRow.appendChild(meta);
      metaRow.appendChild(copyIcon);
    } else {
      metaRow.appendChild(copyIcon);
      if (meta.innerHTML) metaRow.appendChild(meta);
      metaRow.appendChild(time);
    }
    
    row.appendChild(bubble);
    row.appendChild(metaRow);
    messagesDiv.appendChild(row);
    row.scrollIntoView({ behavior: "smooth", block: "end" });

    if (isUser) {
      messageCount++;
      updateSessionInfo();
    }
  }
    
  function showTyping() {
    const t = document.createElement("div");
    t.id = "typing";
    t.className = "message bot";
    t.innerHTML = "<div>Assistant is thinking…</div>";
    messagesDiv.appendChild(t);
    t.scrollIntoView({ behavior: "smooth", block: "end" });
  }



  function hideTyping() {
    const t = document.getElementById("typing");
    if (t) t.remove();
  }

// === UPDATE: sendMessage() — ADD vertical_id to payload ===
  async function sendMessage() {
    if (isProcessing) return;
    const text = input.value.trim();
    if (!text) return;

    // ✅ Vertical already loaded from session
    if (!SELECTED_VERTICAL_ID) {
      addMessage("Session expired. Please login again.", false);
      setTimeout(() => window.location.href = "login.html", 1500);
      return;
    }

    isProcessing = true;
    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;

    addMessage(text, true);
    showTyping();

    try {
      const res = await fetch(`${ACTIVE_CONFIG.BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
          vertical_id: SELECTED_VERTICAL_ID  // ← NEW
        }),
      });
        
        const data = await res.json();
        hideTyping();

        if (res.ok) {
            // Update session
            if (data.session_id) {
                sessionId = data.session_id;
                localStorage.setItem("chatbot_session_id", sessionId);
                updateSessionInfo();
            }
            
            // ✅ FIXED: Handle Excel download without page refresh
            if (data.download_ready && data.report_id) {
                // Show response message
                //addMessage(data.response, false, data);
                
                // Create download button
                const downloadContainer = createDownloadButton(
                    data.report_id,
                    data.download_filename,
                    data.rows,
                    data.file_size
                );
                messagesDiv.appendChild(downloadContainer);
                downloadContainer.scrollIntoView({ behavior: "smooth", block: "end" });
                
                logging.info(`✅ Download button created for: ${data.download_filename}`);
            } else {
                // Normal message
                addMessage(data.response, false, data);
            }
        } else {
            addMessage(` ${data.error || data.response}`, false);
        }
    } catch (err) {
        // hideTyping();
        // addMessage("Sorry, something went wrong. Please try again.", false);
        // console.error("Chat error:", err);
    } finally {
        isProcessing = false;
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

// === UPDATE: Save vertical selection on change (optional) ===


  if (sendBtn && input) {
    sendBtn.addEventListener("click", sendMessage);
    input.addEventListener("keypress", (e) => {
      if (e.key === "Enter") sendMessage();
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", async () => {
      const file = fileInput.files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append("file", file);
      uploadStatus.textContent = "Uploading…";
      uploadStatus.style.color = "#050505ff";
      try {
        const res = await fetch(`${ACTIVE_CONFIG.BASE_URL}/upload_document`, {
          method: "POST",
          body: fd,
        });
        const data = await res.json();
        if (res.ok) {
          uploadStatus.textContent = `✅ ${data.message}`;
          uploadStatus.style.color = "#28a745";
          addMessage(`📄 Document uploaded! ${data.chunks_count} chunks indexed.`, false);
        } else {
          uploadStatus.textContent = `❌ ${data.error}`;
          uploadStatus.style.color = "#dc3545";
        }
      } catch {
        uploadStatus.textContent = "❌ Upload failed";
        uploadStatus.style.color = "#dc3545";
      }
      fileInput.value = "";
      setTimeout(() => (uploadStatus.textContent = ""), 3000);
    });
  }

  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
    // ✅ Clear all session data
    localStorage.removeItem("loggedIn");
    localStorage.removeItem("chatbot_session_id");
    localStorage.removeItem("vertical_id");
    localStorage.removeItem("vertical_name");
    localStorage.removeItem("username");
    
    setTimeout(() => {
      window.location.href = "login.html";
    }, 100);
  });
}

  initSession();

  let inactiveTimer,warningTimer;
  const INACTIVE_TIMER =5*60*1000;
  const WARNING_TIMER=INACTIVE_TIMER-3000;//30 seconds before logout

  function resetInactiveTimer(){
    clearTimeout(inactiveTimer);
    clearTimeout(warningTimer);

    warningTimer=setTimeout(()=>{
      alert("You will be logged out soon due to inactivity")
    },WARNING_TIMER);

    inactiveTimer=setTimeout(()=>{
      localStorage.removeItem("loggedIn");
      localStorage.removeItem("chatbot_session_id");
      window.location.href="login.html";

    },INACTIVE_TIMER);
  }

  ["mousemove","keydown","click","scroll","touchstart"].forEach(evt=>{
    window.addEventListener(evt,resetInactiveTimer);
  });
    
  resetInactiveTimer();
});

// Add this function to detect report requests
function isReportRequest(message) {
    const reportKeywords = ['report', 'excel', 'download', 'export', 'xlsx', 'pdf'];
    const lowerMessage = message.toLowerCase();
    return reportKeywords.some(keyword => lowerMessage.includes(keyword));
}

// Modify your existing chat send function
async function sendMessage() {
    const message = document.getElementById('user-input').value;
    
    if (isReportRequest(message)) {
        // Call download endpoint
        const response = await fetch('/download_report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `chiller_report_${Date.now()}.xlsx`;
            a.click();
            
            // Show success message in chat
            addMessage('assistant', '✅ Report downloaded successfully!');
        } else {
            addMessage('assistant', '❌ Failed to generate report');
        }
    } else {
        // Normal chat flow
        // ... your existing chat code ...
    }
}


function createDownloadButton(reportId, fileName, rows, fileSize) {
// function createDownloadButton(reportId, fileName, rows, fileSize, metadata = {}) {
    const container = document.createElement('div');
    container.className = 'download-container';
    container.style.cssText = `
        margin: 8px 0;
        padding: 12px 7px;
        background: #eef3f0;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 300px;
        animation: fadeIn 0.3s ease;
    `;
    

    // File info - compact
    const info = document.createElement('div');
    info.style.cssText = 'color: #423e3e; flex: 1; font-size: 13px; line-height: 1.4;';
    info.innerHTML = `
        <div style="font-weight: 500;">${fileName}</div>
        <div style="font-size: 11px; opacity: 0.9; margin-top: 2px;">${rows} rows • ${fileSize}</div>
    `;
    
    // ✅ Download ICON instead of button
    const downloadIcon = document.createElement('span');
    downloadIcon.className = 'material-symbols-light--download-sharp';

    downloadIcon.innerHTML = '<span class="iconify" data-icon="material-symbols:download" style="font-size: 24px;"></span>';
    downloadIcon.style.cssText = `
        font-size: 28px;
        color: #423e3e;
        cursor: pointer;
        padding: 8px;
        border-radius: 50%;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.2);
    `;
   // Hover effect
    downloadIcon.addEventListener('mouseenter', () => {
        downloadIcon.style.background = 'rgba(152, 152, 152, 0.3)';
        downloadIcon.style.transform = 'scale(1.1)';
    });
    
    downloadIcon.addEventListener('mouseleave', () => {
        downloadIcon.style.background = ' #eef3f0';
        downloadIcon.style.transform = 'scale(1)';
    });
    
    // Download handler
    downloadIcon.addEventListener('click', async (e) => {
        e.preventDefault();

        const downloadUrl = `${ACTIVE_CONFIG.BASE_URL}/download_excel/${reportId}?filename=${encodeURIComponent(fileName)}`;
        
        // ✅ REMOVED: Don't prevent multiple clicks
        // if (downloadBtn.disabled) return;
        
        try {
            // Show loading state (optional)
            downloadIcon.style.opacity = '0.5';
            downloadIcon.style.cursor = 'wait';
            
            const response = await fetch(downloadUrl);
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            console.log(`✅ Downloaded: ${fileName}`);
            
            // Reset state
            downloadIcon.style.opacity = '1';
            downloadIcon.style.cursor = 'pointer';
            
        } catch (error) {
            console.error('❌ Download failed:', error);
            alert('Download failed. Please try again.');
            
            // Reset state
            downloadIcon.style.opacity = '1';
            downloadIcon.style.cursor = 'pointer';
        }
    });
    
    container.appendChild(info);
    container.appendChild(downloadIcon);
    
    return container;
}


// ✅ Add animation CSS if not already present
if (!document.getElementById('download-animations')) {
    const style = document.createElement('style');
    style.id = 'download-animations';
    style.textContent = `
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(15px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .download-container:hover {
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2) !important;
            transform: translateY(-3px);
        }
        
        .excel-download-btn:active:not(:disabled) {
            transform: scale(0.95) !important;
        }
    `;
    document.head.appendChild(style);
}

async function onVerticalChange(verticalId) {
    const response = await fetch('/load_devices', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vertical_id: verticalId })
    });
    const data = await response.json();
    console.log("Devices loaded:", data.devices);
    // Update UI dropdown, etc.
}


// ✅ Helper: Add console logging for debugging
const logging = {
    info: (msg) => console.log(`[INFO] ${msg}`),
    error: (msg) => console.error(`[ERROR] ${msg}`),
    warn: (msg) => console.warn(`[WARN] ${msg}`)
};