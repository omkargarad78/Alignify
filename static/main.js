// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileList = document.getElementById('fileList');
const fileName = document.getElementById('fileName');
const removeFile = document.getElementById('removeFile');
const jobDescription = document.getElementById('jobDescription');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const analyzeBtnArrow = document.querySelector('#analyzeBtn svg');
const resultsSection = document.getElementById('resultsSection');
const scoreCircle = document.getElementById('scoreCircle');
const scoreValue = document.getElementById('scoreValue');
const emailTemplateBtn = document.getElementById('emailTemplateBtn');
const linkedinMessageBtn = document.getElementById('linkedinMessageBtn');
const messageContent = document.getElementById('messageContent');
const copyBtn = document.getElementById('copyBtn');
const copyTooltip = document.getElementById('copyTooltip');
const templateTitle = document.getElementById('templateTitle');
const missingKeywordsContainer = document.querySelector('.flex.flex-wrap.gap-2');
const suggestionsList = document.querySelector('.space-y-2');
const matchMessage = document.getElementById('matchMessage');
const themeToggle = document.getElementById('themeToggle');

// Variables
let currentFile = null;
let templates = {
  email: '',
  linkedin: ''
};

// Theme functionality
function initTheme() {
  // Check if user preference is stored
  const savedTheme = localStorage.getItem('theme');
  
  if (savedTheme) {
    document.documentElement.className = savedTheme;
  } else {
    // Set dark mode as default
    document.documentElement.className = 'dark';
    localStorage.setItem('theme', 'dark');
  }
}

// Initialize theme on load
initTheme();

// Theme toggle event listener
themeToggle.addEventListener('click', () => {
  if (document.documentElement.classList.contains('dark')) {
    document.documentElement.className = 'light';
    localStorage.setItem('theme', 'light');
  } else {
    document.documentElement.className = 'dark';
    localStorage.setItem('theme', 'dark');
  }
});

// Add animations CSS
const style = document.createElement('style');
style.textContent = `
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @keyframes slideIn {
    from { transform: translateX(-20px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }

  @keyframes popIn {
    0% { transform: scale(0.8); opacity: 0; }
    70% { transform: scale(1.05); }
    100% { transform: scale(1); opacity: 1; }
  }

  .fade-in {
    animation: fadeIn 0.5s ease-out forwards;
  }

  .fade-in-delay-1 {
    animation: fadeIn 0.5s ease-out 0.1s forwards;
    opacity: 0;
  }

  .fade-in-delay-2 {
    animation: fadeIn 0.5s ease-out 0.2s forwards;
    opacity: 0;
  }

  .fade-in-delay-3 {
    animation: fadeIn 0.5s ease-out 0.3s forwards;
    opacity: 0;
  }

  .slide-in {
    animation: slideIn 0.4s ease-out forwards;
  }

  .pop-in {
    animation: popIn 0.5s ease-out forwards;
  }
`;
document.head.appendChild(style);

// Event Listeners
analyzeBtn.addEventListener('click', handleAnalyzeClick);
dropZone.addEventListener('click', () => document.getElementById('fileInput')?.click());
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
removeFile.addEventListener('click', handleRemoveFile);
jobDescription.addEventListener('input', updateCounter);
emailTemplateBtn.addEventListener('click', () => showTemplate('email'));
linkedinMessageBtn.addEventListener('click', () => showTemplate('linkedin'));
copyBtn.addEventListener('click', copyToClipboard);

// If fileInput doesn't exist, create it
if (!document.getElementById('fileInput')) {
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.id = 'fileInput';
  fileInput.accept = '.pdf,.doc,.docx';
  fileInput.style.display = 'none';
  fileInput.addEventListener('change', handleFileInput);
  document.body.appendChild(fileInput);
}

// Event Handler Functions
function handleDragOver(e) {
  e.preventDefault();
  dropZone.classList.add('border-primary-500');
  dropZone.classList.add('bg-primary-50');
  dropZone.classList.add('dark:bg-primary-900/10');
}

function handleDragLeave() {
  dropZone.classList.remove('border-primary-500');
  dropZone.classList.remove('bg-primary-50');
  dropZone.classList.remove('dark:bg-primary-900/10');
}

function handleDrop(e) {
  e.preventDefault();
  handleDragLeave();
  
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
}

function handleFileInput(e) {
  if (e.target.files.length) {
    handleFile(e.target.files[0]);
  }
}

function handleRemoveFile() {
  currentFile = null;
  fileList.classList.add('hidden');
  document.getElementById('fileInput').value = '';
}

function handleAnalyzeClick() {
  if (!currentFile) {
    alert('Please upload a resume file first');
    return;
  }
  
  if (!jobDescription.value.trim()) {
    alert('Please paste a job description');
    return;
  }
  
  // Show loading state
  analyzeBtn.disabled = true;
  analyzeBtn.classList.add('opacity-75');
  analyzeBtn.innerHTML = `
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
    Analyzing
  `;
  
  // Create FormData and send the request
  const formData = new FormData();
  formData.append('resume', currentFile);
  formData.append('job_description', jobDescription.value);
  
  fetch('/analyze', {
    method: 'POST',
    body: formData,
  })
  .then(response => response.json())
  .then(data => {
    console.log('Success:', data);
    displayResults(data);
  })
  .catch(error => {
    console.error('Error:', error);
    alert('An error occurred during analysis. Please try again.');
  })
  .finally(() => {
    // Reset button state
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('opacity-75');
    analyzeBtn.innerHTML = `
      Analyze Match
      <svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"/>
      </svg>
    `;
  });
}

function updateCounter() {
  charCount.textContent = `${jobDescription.value.length} characters`;
}

function displayResults(data) {
  // Store templates
  templates.email = data.email_template || 'No email template generated.';
  templates.linkedin = data.linkedin_message || 'No LinkedIn message generated.';
  
  // Update match score
  const score = Math.round(data.match_score * 100);
  scoreValue.textContent = score;
  
  // Update score circle
  const circumference = 2 * Math.PI * 44; // 2πr
  const offset = circumference - (score / 100) * circumference;
  scoreCircle.style.strokeDasharray = `${circumference} ${circumference}`;
  scoreCircle.style.strokeDashoffset = circumference;
  
  // Set color based on score
  if (score < 40) {
    scoreCircle.style.stroke = '#ef4444'; // red
  } else if (score < 70) {
    scoreCircle.style.stroke = '#f59e0b'; // yellow/amber
  } else {
    scoreCircle.style.stroke = '#10b981'; // green
  }
  
  // Animate the score with a delay
  setTimeout(() => {
    scoreCircle.style.transition = 'stroke-dashoffset 1.5s ease-in-out';
    scoreCircle.style.strokeDashoffset = offset;
  }, 300);
  
  // Update match message
  if (score < 40) {
    matchMessage.textContent = 'Your resume needs improvements to match this job.';
  } else if (score < 70) {
    matchMessage.textContent = 'Your resume is a moderate match for this position.';
  } else {
    matchMessage.textContent = 'Great match! Your resume aligns well with this job.';
  }
  
  // Display missing keywords
  missingKeywordsContainer.innerHTML = '';
  if (data.missing_keywords && data.missing_keywords.length > 0) {
    data.missing_keywords.forEach(keyword => {
      const pill = document.createElement('span');
      pill.className = 'bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 px-2 py-1 rounded-full text-xs font-medium';
      pill.textContent = keyword;
      missingKeywordsContainer.appendChild(pill);
    });
  } else {
    const noneMsg = document.createElement('span');
    noneMsg.className = 'text-gray-500 dark:text-gray-400 text-sm italic';
    noneMsg.textContent = 'No critical keywords missing';
    missingKeywordsContainer.appendChild(noneMsg);
  }
  
  // Display improvement suggestions
  suggestionsList.innerHTML = '';
  if (data.improvement_suggestions && data.improvement_suggestions.length > 0) {
    data.improvement_suggestions.forEach(suggestion => {
      const li = document.createElement('li');
      li.className = 'text-sm text-gray-700 dark:text-gray-300 flex items-start';
      li.innerHTML = `
        <svg class="w-4 h-4 text-primary-500 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
        </svg>
        ${suggestion}
      `;
      suggestionsList.appendChild(li);
    });
  } else {
    const li = document.createElement('li');
    li.className = 'text-sm text-gray-500 dark:text-gray-400 italic';
    li.textContent = 'No specific suggestions available';
    suggestionsList.appendChild(li);
  }
  
  // Show email template by default
  showTemplate('email');
  
  // Show results with animation
  resultsSection.classList.remove('hidden');
  
  // Apply animations to each result section
  const resultSections = resultsSection.querySelectorAll('.bg-white.dark\\:bg-gray-800');
  resultSections.forEach((section, index) => {
    section.classList.add(`fade-in-delay-${index}`);
  });
  
  // Scroll to results
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showTemplate(type) {
  // Update active button
  if (type === 'email') {
    emailTemplateBtn.classList.remove('bg-white', 'dark:bg-gray-700');
    emailTemplateBtn.classList.add('bg-primary-50', 'dark:bg-primary-900/30');
    
    linkedinMessageBtn.classList.remove('bg-primary-50', 'dark:bg-primary-900/30');
    linkedinMessageBtn.classList.add('bg-white', 'dark:bg-gray-700');
    
    templateTitle.textContent = 'Email Template';
  } else {
    linkedinMessageBtn.classList.remove('bg-white', 'dark:bg-gray-700');
    linkedinMessageBtn.classList.add('bg-primary-50', 'dark:bg-primary-900/30');
    
    emailTemplateBtn.classList.remove('bg-primary-50', 'dark:bg-primary-900/30');
    emailTemplateBtn.classList.add('bg-white', 'dark:bg-gray-700');
    
    templateTitle.textContent = 'LinkedIn Message';
  }
  
  // Set template content with animation
  messageContent.style.opacity = 0;
  setTimeout(() => {
    messageContent.textContent = templates[type];
    messageContent.style.transition = 'opacity 0.3s ease-out';
    messageContent.style.opacity = 1;
  }, 200);
}

function copyToClipboard() {
  navigator.clipboard.writeText(messageContent.textContent).then(() => {
    // Show tooltip
    copyTooltip.textContent = 'Copied!';
    copyTooltip.classList.add('opacity-100');
    
    // Hide tooltip after 2 seconds
    setTimeout(() => {
      copyTooltip.classList.remove('opacity-100');
      
      // Reset text after fade out
      setTimeout(() => {
        copyTooltip.textContent = 'Copy to clipboard';
      }, 300);
    }, 2000);
  });
}

// File handling function
function handleFile(file) {
  const validTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
  
  if (!validTypes.includes(file.type)) {
    alert('Please upload a PDF, DOC, or DOCX file');
    return;
  }
  
  currentFile = file;
  fileName.textContent = file.name;
  fileList.classList.remove('hidden');
}