# Preserve paper.pdf as the manuscript output, but use the root document's
# basename for every other document.  A shared job name would make main.tex and
# cover_letter.tex read and overwrite each other's auxiliary files.
my $is_manuscript = grep { m{(?:^|/)main\.tex$} } @ARGV;
$jobname = $is_manuscript ? 'paper' : '%A';
