const proj = "MetPy";

$(document).ready(function() {
    cur_ver = DOCUMENTATION_OPTIONS.VERSION;
    end = cur_ver.lastIndexOf('.');
    if (end > -1) {
        cur_ver = 'v' + cur_ver.substring(0, end);
    }
    console.log('cur_ver: ' + cur_ver);

    $.getJSON('/' + proj + '/versions.json', function(data) {
        if (cur_ver !== data.latest) {
            let msg;
            if (cur_ver.includes('dev') || data.prereleases.indexOf(cur_ver) > -1) {
                msg = 'development / pre-release';
            } else {
                msg = 'previous';
            }
            content = $('<div class="alert alert-warning alert-version" role="alert">This documentation page is for a ' + msg +
                        ' version. For the latest release version, go to <a class="alert-link" href="https://unidata.github.io/MetPy/latest/">https://unidata.github.io/MetPy/latest/</a>');
            $('#banner').append(content);
        }

        $.each(data.versions, function() {
            if (this !== 'latest') {
                const url = DOCUMENTATION_OPTIONS.URL_ROOT + '../' + this;
                const name = this.startsWith('v') ? this.substring(1) : this;
                $('#version-menu').append('<a class="dropdown-item" href="' + url + '">' + name + '</a>');
            }
        });
    });
});

// Borrowed from Bokeh docs to look for an alert.html at the base of the
// docs repo and add that to the banner if present.
$(document).ready(function () {
    $.get('/' + proj + '/alert.html', function (data) {
       if (data.length > 0) {
         content = $('<div class="alert alert-primary alert-news" role="alert">' + data + '</div>')
         $('#banner').prepend(content);
       }
    })
 })
