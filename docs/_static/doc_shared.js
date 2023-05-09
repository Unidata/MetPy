const project = "MetPy";

// Borrowed from Bokeh docs to look for a banner.html at the base of the docs repo and add that
// to the banner if present.
$(document).ready(function () {
    $.get('/' + project + '/banner.html', function (data) {
        if (data.length > 0) {
            console.log(data);
            $('#banner').prepend(data);
        }
    })
 })
