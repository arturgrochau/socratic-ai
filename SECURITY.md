# Security

Socratic AI runs locally and does not accept connections from other machines by default (it binds to `127.0.0.1`). If you deploy it with Docker on a network, put it behind authentication; the `X-User-ID` header is a namespace, not a login.

To report a vulnerability, email the address on the GitHub profile of [arturgrochau](https://github.com/arturgrochau) or open a private security advisory on the repository. Please do not file public issues for vulnerabilities. You will get a reply within a week.
