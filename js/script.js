document.addEventListener('DOMContentLoaded', function() {
	const form = document.querySelector('form');
   	const fileInput = document.getElementById('fileUpload');
	
	form.addEventListener('submit', function(event) {
		if (!fileInput.value) {
           		alert('Please select a file to upload.');
            		event.preventDefault();
            		return;
        	}

		const file = fileInput.files[0];
        	const fileType = file.type;
        	if (fileType !== 'text/csv') {
            		alert('Please upload a CSV file.');
            		event.preventDefault();
            		return;
       	 	}
    	});

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
