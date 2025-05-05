def identity_callback(commit):
    if commit.author_email == b"github@phageghost.net":
        commit.author_email = b"3150660+phageghost@users.noreply.github.com"
    if commit.committer_email == b"github@phageghost.net":
        commit.committer_email = b"3150660+phageghost@users.noreply.github.com"
