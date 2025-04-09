// JavaScript for the feature importance analysis website

document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Add toggle functionality for feature details
    const featureCards = document.querySelectorAll('.feature-card');
    
    featureCards.forEach(card => {
        const detailsElement = card.querySelector('.feature-details');
        const toggleButton = card.querySelector('.toggle-details');
        
        if (toggleButton && detailsElement) {
            // Initially hide details
            detailsElement.style.display = 'none';
            
            toggleButton.addEventListener('click', function() {
                if (detailsElement.style.display === 'none') {
                    detailsElement.style.display = 'block';
                    toggleButton.textContent = 'Hide Details';
                } else {
                    detailsElement.style.display = 'none';
                    toggleButton.textContent = 'Show Details';
                }
            });
        }
    });
    
    // Add active class to navigation links based on scroll position
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('nav a');
    
    window.addEventListener('scroll', function() {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (pageYOffset >= sectionTop - 100) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
});
