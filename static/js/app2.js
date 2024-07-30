var x=document.getElementById('logout');
var display=0;

function hideShow(){
    
    
    if(display==0)
    {
        console.log("function called");
        x.style.display='block';
        display=1;
    }
    else
    {
        x.style.display='none';
        display=0;
    }
    
}

const ques=document.getElementById('ques');

function generateQuestions(){
    ques.style.display='block';
}