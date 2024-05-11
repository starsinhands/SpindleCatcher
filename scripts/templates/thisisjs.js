var upload_btn = document.getElementById("upload_btn");

function upload_btn_onclick(){
	let input = document.getElementById("data_upload");
	input.click();
}
function changeTag(){
	let file_status = document.getElementById("file_status");	
	let input = document.getElementById("data_upload");
	file_status.innerHTML = input.files[0].name;
}
function analyse(){
	const xhr = new XMLHttpRequest();
	let input = document.getElementById("data_upload");
	let file = input.files[0];
	let formdata = new FormData();
	formdata.append("data_file",file);
	xhr.open('POST','/start');
	xhr.send(formdata);
}
