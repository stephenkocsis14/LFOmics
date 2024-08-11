document.addEventListener('DOMContentLoaded', function() {
    // Form Validation
    const form = document.querySelector('form');
    const fileInput = document.getElementById('fileUpload');
    const emailInput = document.getElementById('email');
    
    form.addEventListener('submit', function(event) {
        // Check if a file is selected
        if (!fileInput.value) {
            alert('Please select a file to upload.');
            event.preventDefault();
            return;
        }

        // Check if the selected file is a CSV
        const file = fileInput.files[0];
        const fileType = file.type;
        if (fileType !== 'text/csv') {
            alert('Please upload a CSV file.');
            event.preventDefault();
            return;
        }

        // Check if email is valid
        const email = emailInput.value;
        if (!validateEmail(email)) {
            alert('Please enter a valid email address.');
            event.preventDefault();
            return;
        }
    });

    // Email validation function
    function validateEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(String(email).toLowerCase());
    }

    // Smooth scrolling for anchor links
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(event) {
            event.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            window.scrollTo({
                top: targetElement.offsetTop,
                behavior: 'smooth'
            });
        });
    });
});
