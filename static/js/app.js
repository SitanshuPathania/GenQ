
let mainPage=document.querySelector('main');
let registerLink=document.querySelector('.Register-link');
let loginLink=document.querySelector('.Login-link');

let btnLogin=document.querySelector('.btn');

let iconClose=document.querySelector('.close-icon');

let iconClose2=document.querySelector('.close-icon2');

registerLink.addEventListener('click',function(){
    mainPage.classList.add('active');
});

loginLink.addEventListener('click',function(){
    mainPage.classList.remove('active');
});

btnLogin.addEventListener('click',function(){
    mainPage.classList.add('active-popup');
});

iconClose.addEventListener('click',function(){
    mainPage.classList.remove('active-popup');
});

iconClose2.addEventListener('click',function(){
    mainPage.classList.remove('active-popup');
});

function show()
{
    let hd=document.querySelector('#nothidden');
    hd.style.display='block';
}

function hide()
{
    let nh=document.querySelector('#nothidden');
    nh.style.display='none';
}

let logout=document.querySelector('logout');

function toggleLogout(){
    logout.classList.toggle('toggleLogout');
}



// let user=document.getElementById('user');
// let logout=document.getElementById('logout');

// let disp=0;
// user.addEventListener('click',function(){
//     if(disp==0)
//     {
//         logout.style.display='block';
//         disp=1;
//     }
//     else{
//         logout.style.display='block';
//         disp=0;
//     }
// })



function registeredSuccessfully()
{
    alert("Registered Successfully");
}






