/* --- Premium Interactions and Behavior Script --- */

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. Dark/Light Theme Switcher ---
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const themeToggleText = document.getElementById('theme-toggle-text');
    const htmlElement = document.documentElement;

    // Check for saved user preference or system preference
    const savedTheme = localStorage.getItem('theme') || 'dark';
    htmlElement.setAttribute('data-theme', savedTheme);
    updateThemeUI(savedTheme);

    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = htmlElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        // Apply theme
        htmlElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        updateThemeUI(newTheme);
    });

    function updateThemeUI(theme) {
        if (theme === 'dark') {
            themeToggleText.textContent = 'Modo Claro';
        } else {
            themeToggleText.textContent = 'Modo Oscuro';
        }
    }

    // --- 2. Mobile Sidebar Drawer Controller ---
    const mobileSidebarToggle = document.getElementById('mobile-sidebar-toggle');
    const appSidebar = document.getElementById('app-sidebar');

    mobileSidebarToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        appSidebar.classList.toggle('open');
        
        // Update icon based on state
        const icon = mobileSidebarToggle.querySelector('i');
        if (appSidebar.classList.contains('open')) {
            icon.className = 'fa-solid fa-xmark';
        } else {
            icon.className = 'fa-solid fa-bars';
        }
    });

    // Close sidebar on clicking outside in mobile view
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 1024 && appSidebar.classList.contains('open')) {
            if (!appSidebar.contains(e.target) && e.target !== mobileSidebarToggle) {
                appSidebar.classList.remove('open');
                mobileSidebarToggle.querySelector('i').className = 'fa-solid fa-bars';
            }
        }
    });

    // --- 3. Dynamic Sidebar Navigation highlighting (Scroll Spy) ---
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section');

    const observerOptions = {
        root: null,
        rootMargin: '-20% 0px -60% 0px', // Trigger when section occupies the middle of screen
        threshold: 0
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const activeId = entry.target.getAttribute('id');
                
                // Update nav links
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${activeId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }, observerOptions);

    sections.forEach(section => observer.observe(section));

    // Smooth navigation click behavior
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            // Close mobile menu if open
            if (appSidebar.classList.contains('open')) {
                appSidebar.classList.remove('open');
                mobileSidebarToggle.querySelector('i').className = 'fa-solid fa-bars';
            }
        });
    });

    // --- 4. Dynamic CRISP-ML Tab Controller ---
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active states from buttons
            tabButtons.forEach(btn => {
                btn.classList.remove('active');
                btn.setAttribute('aria-selected', 'false');
            });
            
            // Add active state to clicked button
            button.classList.add('active');
            button.setAttribute('aria-selected', 'true');
            
            // Hide all panels
            tabPanels.forEach(panel => {
                panel.classList.remove('active');
            });
            
            // Show corresponding panel
            const panelId = button.getAttribute('aria-controls');
            const targetPanel = document.getElementById(panelId);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });
});
