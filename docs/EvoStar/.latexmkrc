# 中文稿默认用 XeLaTeX；兼容编辑器自带的 latexmk -pdf 配方。
$pdf_mode = 5;
$xelatex = 'xelatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$pdflatex = 'xelatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
1;
