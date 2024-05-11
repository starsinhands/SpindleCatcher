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
	let loader = document.getElementsByClassName("loader")[0].style.display = "flex";
	let resample_rate = document.getElementById("resample_rate").value;
	let patient_age = document.getElementById("patient_age").value;
	let patient_name = document.getElementById("patient_name").value;
	let patient_gender = document.getElementById("patient_gender").value;
	const xhr = new XMLHttpRequest();
	let input = document.getElementById("data_upload");
	let file = input.files[0];
	let formdata = new FormData();
	formdata.append("data_file",file);
	formdata.append("resample_rate",resample_rate);
	formdata.append("patient_age",patient_age);
	formdata.append("patient_name",patient_name);
	formdata.append("patient_gender",patient_gender);
	xhr.open('POST','/start');
	xhr.send(formdata);
	xhr.onload = function(){
		window.location.href="/result";
		loader.style.display = "None";
	}
}
