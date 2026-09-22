const scenes = {
  "reef-lab": {
    title: "Reef laboratory",
    description: "Complex coral structure with moving fish, illumination variation, and open-water scattering.",
  },
  "fish-reef": {
    title: "Fish reef",
    description: "A textured reef with repeated structure, dense fish motion, and shallow-water color variation.",
  },
  "rock-garden": {
    title: "Rock garden",
    description: "High-frequency rock geometry, schooling fish, and strong natural illumination gradients.",
  },
  "submerged-structure": {
    title: "Submerged structure",
    description: "Low-visibility open water around a large persistent man-made surface.",
  },
  "green-water-rock": {
    title: "Green-water rock",
    description: "Severe color cast and scattering around a compact rocky surface.",
  },
  "shallow-rock": {
    title: "Shallow rock",
    description: "Close-range surface geometry with turbidity, blur, and partial occlusion.",
  },
};

const cloudAlt = {
  "reef-lab": "Dense Water3D point-cloud preview of the reef laboratory scene",
  "green-water-rock": "Dense Water3D point-cloud preview of the green-water rock scene",
  "shallow-rock": "Dense Water3D point-cloud preview of the shallow rock scene",
};

const header = document.querySelector("[data-header]");
const navToggle = document.querySelector("[data-nav-toggle]");
const nav = document.querySelector("[data-nav]");

const updateHeader = () => header?.classList.toggle("is-scrolled", window.scrollY > 24);
updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

navToggle?.addEventListener("click", () => {
  const open = nav.classList.toggle("is-open");
  navToggle.setAttribute("aria-expanded", String(open));
});
nav?.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
  nav.classList.remove("is-open");
  navToggle?.setAttribute("aria-expanded", "false");
}));

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("is-visible");
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

let activeScene = "reef-lab";
let activeFrame = 1;
const sceneImage = document.querySelector("[data-scene-image]");
const sceneIndex = document.querySelector("[data-scene-index]");
const sceneTitle = document.querySelector("[data-scene-title]");
const sceneDescription = document.querySelector("[data-scene-description]");

function renderScene() {
  const details = scenes[activeScene];
  sceneImage?.classList.add("is-changing");
  window.setTimeout(() => {
    if (sceneImage) {
      sceneImage.src = `./static/images/scenes/${activeScene}/frame-${activeFrame}.webp`;
      sceneImage.alt = `${details.title}, frame ${activeFrame}`;
    }
    if (sceneIndex) sceneIndex.textContent = `0${activeFrame} / 03`;
    if (sceneTitle) sceneTitle.textContent = details.title;
    if (sceneDescription) sceneDescription.textContent = details.description;
    window.requestAnimationFrame(() => sceneImage?.classList.remove("is-changing"));
  }, 130);
}

document.querySelectorAll("[data-scene]").forEach((button) => {
  button.addEventListener("click", () => {
    activeScene = button.dataset.scene;
    activeFrame = 1;
    document.querySelectorAll("[data-scene]").forEach((item) => {
      const selected = item === button;
      item.classList.toggle("is-active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    renderScene();
  });
});
document.querySelector("[data-scene-prev]")?.addEventListener("click", () => {
  activeFrame = activeFrame === 1 ? 3 : activeFrame - 1;
  renderScene();
});
document.querySelector("[data-scene-next]")?.addEventListener("click", () => {
  activeFrame = activeFrame === 3 ? 1 : activeFrame + 1;
  renderScene();
});

const cloudImage = document.querySelector("[data-cloud-image]");
document.querySelectorAll("[data-cloud]").forEach((button) => {
  button.addEventListener("click", () => {
    const cloud = button.dataset.cloud;
    document.querySelectorAll("[data-cloud]").forEach((item) => {
      const selected = item === button;
      item.classList.toggle("is-active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    if (cloudImage) {
      cloudImage.src = `./static/images/clouds/${cloud}.webp`;
      cloudImage.alt = cloudAlt[cloud];
    }
  });
});

const lightbox = document.querySelector("[data-lightbox-dialog]");
const lightboxImage = document.querySelector("[data-lightbox-image]");
document.querySelectorAll("[data-lightbox]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!lightbox || !lightboxImage) return;
    lightboxImage.src = button.dataset.lightbox;
    lightbox.showModal();
    document.body.classList.add("is-locked");
  });
});
const closeLightbox = () => {
  lightbox?.close();
  document.body.classList.remove("is-locked");
};
document.querySelector("[data-lightbox-close]")?.addEventListener("click", closeLightbox);
lightbox?.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});
lightbox?.addEventListener("close", () => document.body.classList.remove("is-locked"));
