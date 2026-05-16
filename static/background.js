// 3D background using three.js - lightweight farm scene
let scene, camera, renderer, farmElements = [];
function initBackground(){
    const canvas = document.getElementById('backgroundCanvas');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB);
    camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
    camera.position.set(0,20,60);
    renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(50,100,50);
    scene.add(light);
    const ambient = new THREE.AmbientLight(0xffffff, 0.6); scene.add(ambient);
    const groundGeometry = new THREE.PlaneGeometry(1000,1000);
    const groundMaterial = new THREE.MeshLambertMaterial({ color:0x76b852 });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial); ground.rotation.x = -Math.PI/2; scene.add(ground);
    createClouds(); createTrees(); createBarn();
    function animate(){ requestAnimationFrame(animate); farmElements.forEach(c=>{ if(c.userData.type==='cloud'){ c.position.x += 0.05; if(c.position.x>200) c.position.x=-200; } }); renderer.render(scene,camera); }
    animate();
    window.addEventListener('resize', ()=>{ camera.aspect = window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); });
}
function createClouds(){ for(let i=0;i<6;i++){ const g=new THREE.SphereGeometry(6,8,8); const m=new THREE.MeshLambertMaterial({color:0xffffff}); const cloud=new THREE.Mesh(g,m); cloud.position.set(Math.random()*400-200, 60+Math.random()*20, Math.random()*60-30); cloud.userData.type='cloud'; scene.add(cloud); farmElements.push(cloud);} }
function createTrees(){ for(let i=0;i<10;i++){ const trunk=new THREE.CylinderGeometry(0.8,0.8,6,6); const trunkM=new THREE.MeshLambertMaterial({color:0x8b5a2b}); const t=new THREE.Mesh(trunk,trunkM); t.position.set(Math.random()*300-150,3,Math.random()*200-100); scene.add(t); const leaf=new THREE.SphereGeometry(4,8,8); const leafM=new THREE.MeshLambertMaterial({color:0x2e7d32}); const l=new THREE.Mesh(leaf,leafM); l.position.set(t.position.x,8,t.position.z); scene.add(l);} }
function createBarn(){ const g=new THREE.BoxGeometry(20,12,16); const m=new THREE.MeshLambertMaterial({color:0xa0522d}); const barn=new THREE.Mesh(g,m); barn.position.set(-40,6,-40); scene.add(barn); const siloG=new THREE.CylinderGeometry(3,3,12,12); const siloM=new THREE.MeshLambertMaterial({color:0xd1d5db}); const silo=new THREE.Mesh(siloG,siloM); silo.position.set(-25,6,-42); scene.add(silo); }
window.addEventListener('DOMContentLoaded', initBackground);