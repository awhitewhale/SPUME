const canvas = document.querySelector("[data-point-viewer]");
const viewerStatus = document.querySelector("[data-viewer-status]");

class PointCloudViewer {
  constructor(element) {
    this.canvas = element;
    this.gl = element.getContext("webgl", { antialias: true, alpha: false });
    this.yaw = -0.45;
    this.pitch = -0.18;
    this.zoom = 3.15;
    this.dragging = false;
    this.lastX = 0;
    this.lastY = 0;
    this.pointCount = 0;
    this.lastInteraction = performance.now();
    this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.cache = new Map();

    if (!this.gl) throw new Error("WebGL is not available in this browser.");
    this.setupProgram();
    this.setupEvents();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(element);
    this.resize();
    requestAnimationFrame((time) => this.draw(time));
  }

  shader(type, source) {
    const shader = this.gl.createShader(type);
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      throw new Error(this.gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  setupProgram() {
    const vertex = this.shader(this.gl.VERTEX_SHADER, `
      attribute vec3 aPosition;
      attribute vec3 aColor;
      uniform float uYaw;
      uniform float uPitch;
      uniform float uZoom;
      uniform float uAspect;
      uniform float uPointSize;
      varying vec3 vColor;

      void main() {
        float cy = cos(uYaw), sy = sin(uYaw);
        float cp = cos(uPitch), sp = sin(uPitch);
        vec3 yRot = vec3(
          cy * aPosition.x + sy * aPosition.z,
          aPosition.y,
          -sy * aPosition.x + cy * aPosition.z
        );
        vec3 p = vec3(
          yRot.x,
          cp * yRot.y - sp * yRot.z,
          sp * yRot.y + cp * yRot.z
        );
        float eyeZ = p.z - uZoom;
        float w = max(0.25, -eyeZ);
        float near = 0.1;
        float far = 20.0;
        float a = (far + near) / (near - far);
        float b = (2.0 * far * near) / (near - far);
        gl_Position = vec4(p.x / (0.58 * uAspect), p.y / 0.58, a * eyeZ + b, w);
        gl_PointSize = clamp(uPointSize * 3.2 / w, 1.0, 5.5);
        vColor = aColor;
      }
    `);
    const fragment = this.shader(this.gl.FRAGMENT_SHADER, `
      precision mediump float;
      varying vec3 vColor;
      void main() {
        vec2 point = gl_PointCoord - vec2(0.5);
        if (dot(point, point) > 0.25) discard;
        gl_FragColor = vec4(vColor, 0.96);
      }
    `);
    const program = this.gl.createProgram();
    this.gl.attachShader(program, vertex);
    this.gl.attachShader(program, fragment);
    this.gl.linkProgram(program);
    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      throw new Error(this.gl.getProgramInfoLog(program));
    }

    this.program = program;
    this.buffer = this.gl.createBuffer();
    this.locations = {
      position: this.gl.getAttribLocation(program, "aPosition"),
      color: this.gl.getAttribLocation(program, "aColor"),
      yaw: this.gl.getUniformLocation(program, "uYaw"),
      pitch: this.gl.getUniformLocation(program, "uPitch"),
      zoom: this.gl.getUniformLocation(program, "uZoom"),
      aspect: this.gl.getUniformLocation(program, "uAspect"),
      pointSize: this.gl.getUniformLocation(program, "uPointSize"),
    };
    this.gl.clearColor(1, 1, 1, 1);
    this.gl.enable(this.gl.DEPTH_TEST);
  }

  setupEvents() {
    const interact = () => { this.lastInteraction = performance.now(); };
    this.canvas.addEventListener("pointerdown", (event) => {
      this.dragging = true;
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      this.canvas.setPointerCapture(event.pointerId);
      interact();
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!this.dragging) return;
      this.yaw += (event.clientX - this.lastX) * 0.007;
      this.pitch = Math.max(-1.25, Math.min(1.25, this.pitch + (event.clientY - this.lastY) * 0.007));
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      interact();
    });
    const stop = () => { this.dragging = false; interact(); };
    this.canvas.addEventListener("pointerup", stop);
    this.canvas.addEventListener("pointercancel", stop);
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.zoom = Math.max(1.8, Math.min(6.2, this.zoom + event.deltaY * 0.0025));
      interact();
    }, { passive: false });
    this.canvas.addEventListener("dblclick", () => this.reset());
  }

  reset() {
    this.yaw = -0.45;
    this.pitch = -0.18;
    this.zoom = 3.15;
    this.lastInteraction = performance.now();
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.gl.viewport(0, 0, width, height);
    }
  }

  async load(name) {
    viewerStatus.textContent = "Loading reconstruction…";
    viewerStatus.classList.remove("is-hidden");
    let data = this.cache.get(name);
    if (!data) {
      const response = await fetch(`./static/point-clouds/${name}.spc`);
      if (!response.ok) throw new Error(`Could not load ${name}.`);
      data = new Float32Array(await response.arrayBuffer());
      this.cache.set(name, data);
    }
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW);
    this.pointCount = data.length / 6;
    this.reset();
    viewerStatus.classList.add("is-hidden");
  }

  draw(time) {
    this.resize();
    if (!this.dragging && !this.reducedMotion && time - this.lastInteraction > 900) {
      this.yaw += 0.0018;
    }
    const gl = this.gl;
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (this.pointCount) {
      gl.useProgram(this.program);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.enableVertexAttribArray(this.locations.position);
      gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(this.locations.color);
      gl.vertexAttribPointer(this.locations.color, 3, gl.FLOAT, false, 24, 12);
      gl.uniform1f(this.locations.yaw, this.yaw);
      gl.uniform1f(this.locations.pitch, this.pitch);
      gl.uniform1f(this.locations.zoom, this.zoom);
      gl.uniform1f(this.locations.aspect, this.canvas.width / this.canvas.height);
      gl.uniform1f(this.locations.pointSize, Math.min(window.devicePixelRatio || 1, 2) * 3.0);
      gl.drawArrays(gl.POINTS, 0, this.pointCount);
    }
    requestAnimationFrame((nextTime) => this.draw(nextTime));
  }
}

let viewer;
if (canvas) {
  try {
    viewer = new PointCloudViewer(canvas);
    viewer.load("reef-lab").catch((error) => {
      viewerStatus.textContent = error.message;
    });
  } catch (error) {
    viewerStatus.textContent = error.message;
  }
}

document.querySelectorAll("[data-cloud-source]").forEach((button) => {
  button.addEventListener("click", async () => {
    document.querySelectorAll("[data-cloud-source]").forEach((item) => {
      const selected = item === button;
      item.classList.toggle("is-active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    try {
      await viewer?.load(button.dataset.cloudSource);
    } catch (error) {
      viewerStatus.textContent = error.message;
      viewerStatus.classList.remove("is-hidden");
    }
  });
});

const caseLabels = {
  particles: "particles",
  caustics: "caustics",
  "temporary-occlusion": "temporary occlusion",
};
const comparison = document.querySelector("[data-comparison]");
const comparisonRange = document.querySelector("[data-comparison-range]");
const observation = document.querySelector("[data-observation]");
const result = document.querySelector("[data-result]");

comparisonRange?.addEventListener("input", () => {
  comparison.style.setProperty("--split", `${comparisonRange.value}%`);
});

document.querySelectorAll("[data-case]").forEach((button) => {
  button.addEventListener("click", () => {
    const name = button.dataset.case;
    document.querySelectorAll("[data-case]").forEach((item) => {
      const selected = item === button;
      item.classList.toggle("is-active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    observation.src = `./static/images/comparisons/${name}-observation.webp`;
    result.src = `./static/images/comparisons/${name}-spume.webp`;
    observation.alt = `Underwater observation affected by ${caseLabels[name]}`;
    result.alt = `SPUME reconstruction under ${caseLabels[name]}`;
    comparisonRange.value = 50;
    comparison.style.setProperty("--split", "50%");
  });
});

const lightbox = document.querySelector("[data-lightbox-dialog]");
const lightboxImage = document.querySelector("[data-lightbox-image]");
document.querySelectorAll("[data-lightbox]").forEach((button) => {
  button.addEventListener("click", () => {
    lightboxImage.src = button.dataset.lightbox;
    lightbox.showModal();
  });
});
const closeLightbox = () => lightbox?.close();
document.querySelector("[data-lightbox-close]")?.addEventListener("click", closeLightbox);
lightbox?.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});
