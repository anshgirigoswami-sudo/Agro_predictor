(function(){
  // Three.js ambience scene for subtle glassy background
  const canvas = document.getElementById('webgl-canvas');
  if(!canvas) return;

  const renderer = new THREE.WebGLRenderer({canvas: canvas, antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
  camera.position.set(0, 20, 80);

  // subtle ambient + directional lights for specular highlights
  const ambient = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambient);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(-10, 20, 10);
  scene.add(dir);

  // Create floating low-poly spheres cloud
  const group = new THREE.Group();
  scene.add(group);

  // Read ambience enabled from localStorage (default true)
  let ambienceEnabled = true;
  try{ ambienceEnabled = (localStorage.getItem('ambienceEnabled') !== '0'); }catch(e){}
  const COUNT = ambienceEnabled ? 48 : 0; // reduced count for performance
  const geometry = new THREE.IcosahedronGeometry(1.6, 0);

  // Use MeshPhysicalMaterial for glassy feel
  const material = new THREE.MeshPhysicalMaterial({
    color: 0xf7fbfd,
    metalness: 0.1,
    roughness: 0.05,
    transmission: 0.9,
    transparent: true,
    opacity: 0.85,
    reflectivity: 0.8,
    clearcoat: 0.2,
    clearcoatRoughness: 0.05,
    envMapIntensity: 0.6
  });

  const items = [];
  for(let i=0;i<COUNT;i++){
    const m = new THREE.Mesh(geometry, material);
    const theta = Math.random()*Math.PI*2;
    const phi = Math.acos((Math.random()*2)-1);
    const r = 20 + Math.random()*80;
    m.position.set(Math.cos(theta)*Math.sin(phi)*r, Math.sin(theta)*Math.sin(phi)*r*0.4, Math.cos(phi)*r);
    m.scale.setScalar(0.6 + Math.random()*1.6);
    m.userData = {vx: (Math.random()-0.5)*0.01, vy: (Math.random()-0.5)*0.01, vz: (Math.random()-0.5)*0.01};
    group.add(m);
    items.push(m);
  }

  // mouse interaction
  const mouse = new THREE.Vector2(0,0);
  const cursor = new THREE.Vector2(0,0);
  window.addEventListener('mousemove', (e)=>{
    const x = (e.clientX / window.innerWidth) * 2 - 1;
    const y = -(e.clientY / window.innerHeight) * 2 + 1;
    mouse.set(x,y);
  });

  // ambience toggle button
  const ambienceToggleBtn = document.getElementById('ambienceToggle');
  if(ambienceToggleBtn){
    const setState = (enabled)=>{
      ambienceEnabled = !!enabled;
      try{ localStorage.setItem('ambienceEnabled', enabled ? '1' : '0'); }catch(e){}
      // if disabled, hide group
      group.visible = !!enabled;
    };
    ambienceToggleBtn.addEventListener('click', ()=> setState(!ambienceEnabled));
    // initialize
    setState(ambienceEnabled);
  }

  // resize handling
  function resize(){
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w,h);
    camera.aspect = w/h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize, {passive:true});
  resize();

  // LERP helper
  function lerp(a,b,t){return a + (b-a)*t}

  // tilt parallax for UI cards (applied in animate loop using smoothed cursor)
  const perspective = document.getElementById('ui-perspective');
  const tiltElements = document.querySelectorAll('.glass-card, .card-floating');

  // animate
  const clock = new THREE.Clock();
  function animate(){
    const t = clock.getElapsedTime();

    // smooth cursor LERP towards mouse (luxury smoothing)
    cursor.x = lerp(cursor.x, mouse.x, 0.08);
    cursor.y = lerp(cursor.y, mouse.y, 0.08);

    // gentle orbit of group
    group.rotation.y += 0.0005;
    group.rotation.x = Math.sin(t*0.05)*0.02;

    // items float and respond slightly to cursor
    items.forEach((it, idx)=>{
      it.position.x += it.userData.vx + (cursor.x*0.2)/(10+idx%10);
      it.position.y += it.userData.vy + (cursor.y*0.2)/(10+idx%10);
      it.rotation.x += 0.002*(idx%5+1);
      it.rotation.y += 0.001*(idx%7+1);
    });

    // micro-parallax card tilt/shadow using smoothed cursor
    if(perspective && tiltElements.length){
      // cursor ranges roughly -1..1; map to small rotations
      const rotX = (-cursor.y) * 6; // degrees
      const rotY = (cursor.x) * 6;
      const shadowX = Math.round(cursor.x * 18);
      const shadowY = Math.round(cursor.y * -18);
      tiltElements.forEach((el, i)=>{
        // subtle depth scaling based on index to layer cards
        const depth = 6 + (i % 3) * 4;
        el.style.transform = `translateZ(${depth}px) rotateX(${rotX}deg) rotateY(${rotY}deg)`;
        // dynamic shadow shift
        el.style.boxShadow = `${shadowX}px ${shadowY}px 40px rgba(2,6,23,0.08), 0 30px 80px rgba(2,6,23,0.12)`;
      });
    }

    // subtle camera parallax based on cursor
    camera.position.x = lerp(camera.position.x, cursor.x*40, 0.02);
    camera.position.y = lerp(camera.position.y, cursor.y*-20 + 20, 0.02);
    camera.lookAt(0,0,0);

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  // start the loop
  requestAnimationFrame(animate);
})();
