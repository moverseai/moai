__SYMBOL_MAP__ = {
    # '!': '%21',
    "#": "%23",
    "$": "%24",
    "%": "%25",
    "&": "%26",
    "'": "%27",
    "(": "%28",
    ")": "%29",
    "*": "%2A",
    "+": "%2B",
    ",": "%2C",
    "/": "%2F",
    ":": "%3A",
    ";": "%3B",
    "=": "%3D",
    "?": "%3F",
    "@": "%40",
    "[": "%5B",
    "]": "%5D",
    " ": "%20",
    "<": "%3C",
    ">": "%3E",
    "\\": "%5C",
    "{": "%7B",
    "}": "%7D",
}

__SIZE_MAP__ = {
    "large": "%5Clarge",
    "vlarge": "%5CLarge",
    "vvlarge": "%5CLARGE",
    "huge": "%5Chuge",
    "vhuge": "%5CHuge",
}

# NOTE: https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b

if __name__ == "__main__":
    wing_loss = r"\begin{equation} wing(x) = \left\{ \begin{array}{ll}w \ln (1 + |x|/\epsilon)  & \text{if } |x| < w \\|x| - C  & \text{otherwise}\end{array}\right.\end{equation}"

    adaptive_wing_loss = r"\begin{equation} \small AWing(y,\!\hat{y})\! =\! \begin{cases}  \omega\! \ln(1\! +\!  \displaystyle |\frac{y\!-\!\hat{y}\!}{\epsilon}|^{\alpha-y})\! &\! \text{if } |(y\!-\!\hat{y})|\! <\! \theta   \\   A|y-\hat{y}\!| - C & \text{otherwise} \end{cases} \end{equation}"

    soft_wing_loss = r"\begin{equation}\mathrm{Wing}(x)=\left\{ \begin{array}{ll} \omega \ln({1+\frac{|x|}{\epsilon}})& \mathrm{if}\ |x| < \omega \\ |x| - C  &\mathrm{otherwise} \end{array} \right. \end{equation}"

    berhu_loss = r"\begin{equation}\mathcal{B}(x) = \begin{cases} |x| & |x| \leq c, \\ \frac{x^2 + c^2}{2c} & |x| > c. \\ \end{cases} \end{equation}"

    berhu_threshold = r"c = \frac{1}{5} \max_i(|\tilde{y}_i - y_i|)"

    std_kl_loss = r"\begin{equation}\mathrm{StandardKL}(\mu,\sigma) = \frac{1}{2} \displaystyle\sum_{i} (1+\log(\sigma_{i}^{2}) -\mu_{i}^{2}-\sigma_{i}^{2})\end{equation}"

    std_kl_beta_loss = r"\begin{equation}\beta\mathrm{-StandardKL}(\mu,\sigma) = \beta \, \, \frac{1}{2} \displaystyle\sum_{i} (1+\log(\sigma_{i}^{2}) -\mu_{i}^{2}-\sigma_{i}^{2})\end{equation}"

    std_kl_capacity_loss = r"\begin{equation}\mathrm{CapacityStandardKL}(\mu,\sigma) = \beta \, \, \, |\,\frac{1}{2} \displaystyle\sum_{i} (1+\log(\sigma_{i}^{2}) -\mu_{i}^{2}-\sigma_{i}^{2})\,-\,C\,|\end{equation}"

    std_kl_robust_loss = r"\begin{equation}\mathrm{RobustStandardKL}(\mu,\sigma) = \sqrt{1 + \big(\frac{1}{2} \displaystyle\sum_{i} (1+\log(\sigma_{i}^{2}) -\mu_{i}^{2}-\sigma_{i}^{2})\big)^2} - 1\end{equation}"

    geodesic_loss = r"\begin{equation}\mathcal{d}(R_1,R_2) = \arccos\frac{trace(R_1R_2^T) - 1}{2}\end{equation}"

    output_string = geodesic_loss

    size = "huge"

    for k, v in __SYMBOL_MAP__.items():
        output_string = output_string.replace(k, v)

    url = "https://render.githubusercontent.com/render/math?math="
    print(url + __SIZE_MAP__[size] + output_string)
