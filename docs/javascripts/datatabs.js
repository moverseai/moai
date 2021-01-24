///tab incremental id// 
function setID(){
    var tabs = document.getElementsByClassName("tabbed-set");
    for (var i = 0; i < tabs.length; i++) {
        children = tabs[i].children;
        var counter = 0;
        var iscontent = 0;
        for(var j = 0; j < children.length;j++){
            if(typeof children[j].htmlFor === 'undefined'){
                if((iscontent + 1) % 2 == 0){
                    //check if it is content
                    if(iscontent == 1){
                        btn = children[j].childNodes[1].getElementsByTagName("button");
                    }
                }
                else{
                //if not change the id
                children[j].id = "__tabbed_" + String(i + 1) + "_" + String(counter + 1);
                children[j].name = "__tabbed_" + String(i + 1) //+ "_" + String(counter + 1);
                //make default tab open
                if(j == 0)
                    children[j].click();
                }
                iscontent++;
            }
            else{
                //link to the correct tab
                children[j].htmlFor = "__tabbed_" + String(i+1) + "_" + String(counter + 1);
                counter ++;
            }
        }
    }
}
setID();