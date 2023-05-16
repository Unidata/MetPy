const project = "MetPy";

function documentReady(callback) {
  if (document.readyState != "loading") callback();
  else document.addEventListener("DOMContentLoaded", callback);
}

documentReady(function () {
  // Use PST version metadata to match current doc version
  const cur_ver = DOCUMENTATION_OPTIONS.theme_switcher_version_match;
  console.log("cur_ver: " + cur_ver);

  fetch("/" + project + "/pst-versions.json")
    .then(function (response) {
      return response.json();
    })
    .then(function (data) {
      // Find matching version entry in PST version list
      const entry = data[data.findIndex((obj) => obj.version == cur_ver)];

      // Find out if matched version is latest
      // and construct alert message
      if (entry.is_latest != true) {
        let rel_type;
        if (cur_ver.includes("dev") || entry.is_prerelease == true) {
          rel_type = "development/pre-release";
        } else {
          rel_type = "previous";
        }

        let msg =
          `This documentation page is for a ${rel_type} version. For the latest release version, go to <a class="alert-link" href="https://unidata.github.io/MetPy/latest/">https://unidata.github.io/MetPy/latest/</a>`;

        // Create alert div and fill with message content
        let content = document.createElement("div");
        content.classList.add("alert", "alert-secondary", "alert-version");
        content.setAttribute("role", "alert");
        content.innerHTML = msg;

        // Append alert div to banner div under navbar
        document.querySelector("#banner").appendChild(content);
      } else {
        console.log("MetPy version latest.");
      }
    })
    .catch(function (err) {
      console.warn("Something went wrong.", err);
    });
});

documentReady(function () {
  fetch("/" + project + "/banner.html")
    .then(function (response) {
      return response.text();
    })
    .then(function (html) {
      // If any banner.html exists, parse it and add to banner div
      if (html.length > 0) {
        let parser = new DOMParser();
        let doc = parser.parseFromString(html.trim(), "text/html");

        // Get all div elements from banner.html
        // and prepend them to banner div under navbar
        let divs = doc.getElementsByTagName("div");
        for (let div of divs) {
          document.querySelector("#banner").prepend(div);
        }
      } else {
        console.log("Banner empty.");
      }
    })
    .catch(function (err) {
      return console.warn("Something went wrong.", err);
    });
});
