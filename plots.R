#install.packages("ggplot2")
#install.packages("VGAM")
#install.packages("httpgd")
library("VGAM")
library("ggplot2")
library("latex2exp")



#load data
data_raw = read.csv("C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/original_data/adult/adult.csv")
bin_size = 1
p1 <- ggplot(data_raw, aes(x = age)) +
    geom_histogram(binwidth=bin_size, color='#e9ecef', fill = "#56B4E9", alpha=0.5, position='identity') +
    #aes(y=(1/bin_size)*stat(count)/sum(stat(count))) +
    theme_bw() +
    theme(text = element_text(size=16)) +
    labs(x="age", y="count") +
    xlim(0,100)
print(p1)
ggsave(filename = "age.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)

#load data
bin_size = 1
df = data_raw
df[df == " Outlying-US(Guam-USVI-etc)"] <- " Outlying-US"
p2 <- ggplot(df, aes(x = native.country)) +
    geom_bar(color='#e9ecef', fill = "#56B4E9", alpha=0.5, position='identity') +
    theme_bw() +
    theme(text = element_text(size=16)) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(x="native-country", y="count")  
print(p2)
ggsave(filename = "native_country.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)


#load data
bin_size = 10000
p2 <- ggplot(data_raw, aes(x = fnlwgt)) +
    geom_histogram(binwidth=bin_size, color='#e9ecef', fill = "#56B4E9", alpha=0.5, position='identity') +
    #aes(y=(1/bin_size)*stat(count)/sum(stat(count))) +
    theme_bw() +
    theme(text = element_text(size=16)) +
    labs(x="fnlwgt", y="count") 
print(p2)
ggsave(filename = "fnlwgt.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)


#load data
bin_size = 100
p2 <- ggplot(data_raw, aes(x = capital.loss)) +
    geom_histogram(binwidth=bin_size, color='#e9ecef', fill = "#56B4E9", alpha=0.5, position='identity') +
    #aes(y=(1/bin_size)*stat(count)/sum(stat(count))) +
    theme_bw() +
    theme(text = element_text(size=16)) +
    labs(x="capital-loss", y="count") 
print(p2)
ggsave(filename = "capital-loss.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)

#load data
bin_size = 100
p2 <- ggplot(data_raw, aes(x = capital.loss)) +
    geom_histogram(binwidth=bin_size, color='#e9ecef', fill = "#56B4E9", alpha=0.5, position='identity') +
    theme_bw() +
    theme(text = element_text(size=16)) +
    labs(x="capital-loss > 0", y="count")  +
    xlim(1,4000) 
print(p2)
ggsave(filename = "capital-loss_g0.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)


#load data
bin_size = 1
p2 <- ggplot(data_raw, aes(x = hours.per.week)) +
    geom_histogram(binwidth=bin_size, color='#e9ecef', fill = "#56B4E9", alpha=0.5, position='identity') +
    theme_bw() +
    theme(text = element_text(size=16)) +
    labs(x="hours-per-week", y="count")  
print(p2)
ggsave(filename = "hours_per_week.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)



#plot vanishing gradient
x_train = rnorm(1000, mean = 4, sd = 1)
type = rep("real", 1000)
x_GAN = rnorm(1000, mean = -4, sd = 1)
type2 = rep("fake", 1000)
type = c(type, type2)
x_tot = c(x_train, x_GAN)
z = seq( -8, 8, length.out = 2000)
fz = (bin_size)*exp(5*z)/(1+exp(5*z))
bin_size = 0.3
df = data.frame(x = x_tot, type = type, z, fz)
p <- ggplot(df, aes(x = x, fill = type)) +
      geom_histogram(binwidth=bin_size, color='#e9ecef', alpha=0.5, position='identity') +
      geom_line(aes(x = z, y = fz)) +
      aes(y=(1/bin_size)*stat(count)/sum(stat(count))) +
      scale_fill_manual(name="sample density",values=c("#56B4E9","#D55E00"),labels=unname(TeX(c("$p_G$", "$p$")))) +
        scale_y_continuous(
        # Features of the first axis
        name = "density",
        # Add a second axis and specify its features
        sec.axis = sec_axis(~.*(1/bin_size), name="A(x)")) + 
      theme_bw() +
      theme(text = element_text(size=16)) +
      labs(x="x")
print(p)
ggsave(filename = "van_grad.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)


#plot mode collapse
x1 = rnorm(500, mean = 4, sd = 1)
x2 = rnorm(500, mean = -4, sd = 1)
x_train = c(x1, x2)
type = rep("real", 1000)
x_GAN = rnorm(1000, mean = 4, sd = 1)
type2 = rep("fake", 1000)
type = c(type, type2)
x_tot = c(x_train, x_GAN)
df = data.frame(x = x_tot, type = type)
p <- ggplot(df, aes(x = x, fill = type)) +
      geom_histogram(binwidth=bin_size, color='#e9ecef', alpha=0.5, position='identity') +
      aes(y=(1/bin_size)*stat(count)/sum(stat(count))) +
      scale_fill_manual(name="sample density",values=c("#56B4E9","#D55E00"),labels=unname(TeX(c("$p_G$", "$p$")))) +
      theme_bw() +
      theme(text = element_text(size=16)) +
      labs(x="x", y="density")
print(p)
ggsave(filename = "mode_collapse.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)


#plot vgm
x1 = rnorm(500, mean = 0, sd = 1)
x2 = rnorm(200, mean = 3, sd = 2)
x3 = rnorm(300, mean = -2, sd = 0.4)
x_tot = c(x1, x2, x3)
bin_size = 0.3
norm1 = (0.5*dnorm(x_tot, mean = 0, sd = 1))
norm2 = (0.2*dnorm(x_tot, mean = 3, sd = 2))
norm3 = (0.3*dnorm(x_tot, mean = -2, sd = 0.4))
normTotal = norm1+norm2+norm3
df = data.frame(x = x_tot, norm1 = norm1, norm2 = norm2, norm3 = norm3, normTotal = normTotal)
p <- ggplot(df, aes(x=x)) + 
  geom_histogram(binwidth=bin_size, color = "gray", fill = "darkgray")+#,color = "black", fill = "white") +
  geom_line(aes(y = norm1), color = "blue", size = 0.5) + 
  geom_line(aes(y = norm2), color = "blue", size = 0.5) + 
  geom_line(aes(y = norm3), color = "blue", size = 0.5) + 
  geom_line(aes(y = normTotal), color = "blue",linetype="twodash", size = 1.3) + 
  theme_bw() +
  aes(y=(1/bin_size)*stat(count)/sum(stat(count))) +
  theme(text = element_text(size=16)) + 
  labs(x="x", y="density")
print(p)
ggsave(filename = "MSN.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)









#data
eps <- 0.5
mu <- 0.0936865
#mu = 0.18197481

x <- seq(-25, 25, 0.1)
normal <- dnorm(x, mean = 0, sd  = 1 / mu)
laplace <- dlaplace(x, location = 0, scale = 1 / eps)

#dataframe
df <- data.frame(x = x, normal = normal, laplace = laplace)

#plot
p1 <- ggplot(df, aes(x = x)) +
  geom_line(aes(y = normal, color = "Normal"),
   linetype = "twodash", size = 1.3) +
  geom_line(aes(y = laplace, color = "Laplace"), size = 1.3) +
  theme_bw() +
  theme(text = element_text(size=22)) + 
  labs(x = "value",
       y = "density",
       color = "Legend") +
  scale_color_manual(values = c("darkred", "steelblue")) + 
  guides(color = guide_legend(override.aes = list(linetype = c("solid", "twodash"))))
ggsave(filename = "plotk1.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)

#data
k = 100
x <- seq(-1000, 1000, 0.1)
normal <- dnorm(x, mean = 0, sd  = sqrt(k) / (mu))
laplace <- dlaplace(x, location = 0, scale = k / eps)

#dataframe
df <- data.frame(x = x, normal = normal, laplace = laplace)

#plot
p2 <- ggplot(df, aes(x=x)) + 
  geom_line(aes(y = normal, color = "Normal"),
   linetype = "twodash", size = 1.3) +
  geom_line(aes(y = laplace, color="Laplace"), size = 1.3) +
  theme_bw() +
  theme(text = element_text(size=22)) +
  labs(x = "value",
       y = "density",
       color = "Legend") +
  scale_color_manual(values = c("darkred", "steelblue")) +
  guides(color = guide_legend(override.aes = list(linetype = c("solid", "twodash"))))
ggsave(filename = "plotk100.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)

#data
k = 25
x <- seq(-300, 300, 0.1)
normal <- dnorm(x, mean = 0, sd  = sqrt(k) / (mu))
laplace <- dlaplace(x, location = 0, scale = k / eps)

#dataframe
df <- data.frame(x = x, normal = normal, laplace = laplace)

#plot

p3 <- ggplot(df, aes(x=x)) + 
  geom_line(aes(y = normal, color = "Normal"),
   linetype = "twodash", size = 1.3) +
  geom_line(aes(y = laplace, color="Laplace"), size = 1.3) +
  theme_bw() +
  theme(text = element_text(size=22)) +
  labs(x = "value",
       y = "density",
       color = "Legend") +
  scale_color_manual(values = c("darkred", "steelblue")) +
  guides(color = guide_legend(override.aes = list(linetype = c("solid", "twodash"))))
ggsave(filename = "plotk25.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)


#data
x <- seq(-3.5, 5.0, 0.01)
normal0 <- dnorm(x, mean = 0, sd  = 1)
normal1 <- dnorm(x, mean = 0.1, sd  = 1)
normal2 <- dnorm(x, mean = 0.5, sd  = 1)
normal3 <- dnorm(x, mean = 1.0, sd  = 1)
normal4 <- dnorm(x, mean = 3.0, sd  = 1)

#dataframe
df <- data.frame(x = x, normal = normal1, normal2 = normal2)

#plot
p4 <- ggplot(df, aes(x = x)) +
  geom_line(aes(y = normal0, color = "μ = 0.0"), linetype = "solid", size = 2.0) +
  geom_line(aes(y = normal1, color = "μ = 0.1"), linetype = "twodash", size = 1.2) +
  geom_line(aes(y = normal2, color = "μ = 0.5"), linetype = "dotdash",  size = 1.2) +
  geom_line(aes(y = normal3, color = "μ = 1.0"), linetype = "longdash",  size = 1.2) +
  geom_line(aes(y = normal4, color = "μ = 3.0"), linetype = "dashed", size = 1.2) +
  theme_bw() +
  theme(text = element_text(size=22)) +
  labs(x = "value",
       y = "density",
       color = "Legend") +
  scale_color_manual(values = c("#000000", "darkred", "blue", "#016901", "#f68c00")) + 
  guides(color = guide_legend(override.aes = list(linetype = c("solid", "twodash", "dotdash", "longdash", "dashed")))) 
ggsave(filename = "shiftedNormal.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)




#data
x <- seq(0, 1, 0.001)
G0 <- pnorm(qnorm(1-x) - 0)
G1 <- pnorm(qnorm(1-x) - 0.1)
G2 <- pnorm(qnorm(1-x) - 0.5)
G3 <- pnorm(qnorm(1-x) - 1.0)
G4 <- pnorm(qnorm(1-x) - 3.0)

#dataframe
df <- data.frame(x = x, G0 = G0, G1 = G1, G2 = G2, G3 = G3, G4 = G4)

#plot
p4 <- ggplot(df, aes(x = x)) +
  geom_line(aes(y = G0, color = "μ = 0.0"), linetype = "solid", size = 2.0) +
  geom_line(aes(y = G1, color = "μ = 0.1"), linetype = "twodash", size = 1.2) +
  geom_line(aes(y = G2, color = "μ = 0.5"), linetype = "dotdash",  size = 1.2) +
  geom_line(aes(y = G3, color = "μ = 1.0"), linetype = "longdash",  size = 1.2) +
  geom_line(aes(y = G4, color = "μ = 3.0"), linetype = "dashed", size = 1.2) +
  theme_bw() +
  theme(text = element_text(size=22)) +
  labs(x = "Type I error",
       y = "Type II error",
       color = "Legend") +
  scale_color_manual(values = c("#000000", "darkred", "blue", "#016901", "#f68c00")) + 
  guides(color = guide_legend(override.aes = list(linetype = c("solid", "twodash", "dotdash", "longdash", "dashed")))) +
  xlim(0, 1) +
  ylim(0, 1)
ggsave(filename = "fdp.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)

#plot vgm
x1 = rnorm(100, mean = 0, sd = 1)
x2 = rnorm(200, mean = 1, sd = 2)
x3 = rnorm(100, mean = -1, sd = 0.5)

library("ggplot2")
#plot vgm
set.seed(1234)
x1 = rnorm(2500, mean = 0.5, sd = 1.00)
x2 = rnorm(1000, mean = 3, sd = 2.0)
x3 = rnorm(1500, mean = -2, sd = 0.5)
x_tot = c(x1, x2, x3)
bin_size = 0.3
norm1 = (0.5*dnorm(x_tot, mean = 0.5, sd = 1.00))
norm2 = (0.2*dnorm(x_tot, mean = 3, sd = 2.0))
norm3 = (0.3*dnorm(x_tot, mean = -2, sd = 0.5))
normTotal = norm1+norm2+norm3
df = data.frame(x = x_tot, norm1 = norm1, norm2 = norm2, norm3 = norm3, normTotal = normTotal)
p <- ggplot(df, aes(x=x)) + 
  geom_histogram(binwidth=bin_size, color = "gray", fill = "darkgray")+#,color = "black", fill = "white") +
  geom_line(aes(y = norm1), color = "blue", size = 0.5) + 
  geom_line(aes(y = norm2), color = "blue", size = 0.5) + 
  geom_line(aes(y = norm3), color = "blue", size = 0.5) + 
  geom_line(aes(y = normTotal), color = "blue",linetype="twodash", size = 1.3) + 
  theme_bw() +
  aes(y=(1/bin_size)*stat(count)/sum(stat(count))) +
  theme(text = element_text(size=16)) + 
  coord_cartesian(xlim=c(-4,8), ylim = c(0.0,0.25)) +
  labs(x="x", y="density")
print(p)
ggsave(filename = "MSN.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)



#plot vgm
set.seed(1234)
x1 = rnorm(2500, mean = 0.5, sd = 1.00)
x2 = rnorm(1000, mean = 3, sd = 2.0)
x3 = rnorm(1500, mean = -2, sd = 0.5)
x_tot = c(x1, x2, x3)
bin_size = 0.3
norm1 = (0.5*dnorm(x_tot, mean = 0.5, sd = 1.00))
norm2 = (0.2*dnorm(x_tot, mean = 3, sd = 2.0))
norm3 = (0.3*dnorm(x_tot, mean = -2, sd = 0.5))
normTotal = norm1+norm2+norm3
df = data.frame(x = x_tot, norm1 = norm1, norm2 = norm2, norm3 = norm3, normTotal = normTotal)
p <- ggplot(df, aes(x=x)) + 
  geom_line(aes(y = norm1), color = "blue", size = 0.5) + 
  geom_line(aes(y = norm2), color = "blue", size = 0.5) + 
  geom_line(aes(y = norm3), color = "blue", size = 0.5) +
  geom_segment(aes(x = 0.5, y = -0.01, xend = 0.5, yend = 0.2), linetype = "dashed", color = "#157595", size = 0.5) +
  geom_segment(aes(x = -2, y = -0.01, xend = -2, yend = 0.23), linetype = "dashed", color = "#157595", size = 0.5) +
  geom_segment(aes(x = 3, y = -0.01, xend = 3, yend = 0.04), linetype = "dashed", color = "#157595", size = 0.5) +
  geom_segment(aes(x = -1.35, y = -0.01, xend = -1.35, yend = 0.11), linetype = "longdash", color = "#D55E00", size = 0.5) +
  geom_segment(aes(x = -4.5, y = 0.04, xend = -1.35, yend = 0.04), linetype = "twodash", color = "#E69F00", size = 0.5) +
  geom_segment(aes(x = -4.5, y = 0.005, xend = -1.35, yend = 0.005), linetype = "twodash",color = "#E69F00", size = 0.5) +
  geom_segment(aes(x = -4.5, y = 0.11, xend = -1.35, yend = 0.11), linetype = "twodash",color = "#E69F00", size = 0.5) +
  annotate('text', x = -2, y = -0.02, label = "mu[1]",parse = TRUE,size=6) +
  annotate('text', x = 0.5, y = -0.02, label = "mu[2]",parse = TRUE,size=6) +
  annotate('text', x = 3, y = -0.02, label = "mu[3]",parse = TRUE,size=6) +
  annotate('text', x = -1.33, y = -0.018, label = "x^{(i)}",parse = TRUE,size=6) +
  annotate('text', x = -4.85, y = 0.010, label = "rho[3]",parse = TRUE,size=6) +
  annotate('text', x = -4.85, y = 0.04, label = "rho[2]",parse = TRUE,size=6) +
  annotate('text', x = -4.85, y = 0.115, label = "rho[1]",parse = TRUE,size=6) +
  coord_cartesian(xlim=c(-4,8), ylim = c(0.0,0.25), clip = "off") +
  theme_bw() +
  theme(text = element_text(size=24)) + 
  aes(y=(1/bin_size)*stat(count)/sum(stat(count))) +
  #xlim(-3,9) +
  #ylim(0,0.3)+
  labs(x="x", y="density")
print(p)
ggsave(filename = "MSN2.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)

library("ggplot2")
#plot vgm



#plot vgm
set.seed(1234)
x = seq(0,1,0.001)
density = dbeta(x, shape1 = 6, shape2 = 15)

df = data.frame(x = x, density = density)
p <- ggplot(df, aes(x=x)) +
  geom_line(aes(y = density), size = 1.3) +
  theme_bw() +
  theme(text = element_text(size=22)) +
  geom_segment(aes(x = 0.5, y = 0.0, xend = 0.5, yend = (1/0.5)*pbeta(0.5,6,15)), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = 0.0, y = 0.0, xend = 0.0, yend = (1/0.5)*pbeta(0.5,6,15)), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = 1.0, y = 0.0, xend = 1.0, yend = (1/0.5)*(1-pbeta(0.5,6,15))), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = 0.0, y = (1/0.5)*pbeta(0.5,6,15), xend = 0.5, yend = (1/0.5)*(pbeta(0.5,6,15))), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = 0.5, y = (1/0.5)*(1-pbeta(0.5,6,15)), xend = 1.0, yend = (1/0.5)*(1-pbeta(0.5,6,15))), linetype = "solid", color = "#157595", size = 0.9)
print(p)
ggsave(filename = "binsplit1.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)



#plot 2
one = pbeta(0.25,6,15)
two = pbeta(0.5,6,15) - one
three = 1 - one - two
one = (1/0.25)* one
two = (1/0.25) * two
three = (1/0.5) * three
p <- ggplot(df, aes(x=x)) +
  geom_line(aes(y = density), size = 1.3) +
  theme_bw() +
  theme(text = element_text(size=22)) +
  geom_segment(aes(x = 0.0, y = 0.0, xend = 0.0, yend = one), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = 0.25, y = 0.0, xend = 0.25, yend = max(one,two)), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = 0.5, y = 0.0, xend = 0.5, yend = max(two,three)), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = 1.0, y = 0.0, xend = 1.0, yend =  three), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = 0.0, y = one, xend = 0.25, yend = one), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = 0.25, y = two, xend = 0.5, yend = two), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = 0.5, y = three, xend = 1.0, yend = three), linetype = "solid", color = "#157595", size = 0.9)
print(p)
ggsave(filename = "binsplit2.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)


#plot 3
splits = c(0.0,0.125,0.25,0.375,0.5,1.0)
cdf = (pbeta(splits[2:6], 6, 15) - pbeta(splits[1:5], 6, 15)) * (1.0 / (splits[2:6] - splits[1:5]))

p <- ggplot(df, aes(x=x)) +
  geom_line(aes(y = density), size = 1.3) +
  theme_bw() +
  theme(text = element_text(size=22)) +
  geom_segment(aes(x = splits[1], y = 0.0, xend =splits[1], yend = cdf[1]), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[2], y = 0.0, xend = splits[2], yend = max(cdf[1],cdf[2])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[3], y = 0.0, xend = splits[3], yend = max(cdf[2],cdf[3])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[4], y = 0.0, xend = splits[4], yend = max(cdf[3],cdf[4])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[5], y = 0.0, xend = splits[5], yend = max(cdf[4],cdf[5])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[6], y = 0.0, xend = splits[6], yend = cdf[5]), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[1], y = cdf[1], xend = splits[2], yend = cdf[1]), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[2], y = cdf[2], xend = splits[3], yend = cdf[2]), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[3], y = cdf[3], xend = splits[4], yend = cdf[3]), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[4], y = cdf[4], xend = splits[5], yend = cdf[4]), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[5], y = cdf[5], xend = splits[6], yend = cdf[5]), linetype = "solid", color = "#157595", size = 0.9) 
print(p)
ggsave(filename = "binsplit3.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)



#plot 4
splits = c(0.0,0.125,3/16,0.25,5/16,0.375,7/16,0.5,1.0)
cdf = (pbeta(splits[2:9], 6, 15) - pbeta(splits[1:8], 6, 15)) * (1.0 / (splits[2:9] - splits[1:8]))

p <- ggplot(df, aes(x=x)) +
  geom_line(aes(y = density), size = 1.3) +
  theme_bw() +
  theme(text = element_text(size=22)) +
  geom_segment(aes(x = splits[1], y = 0.0, xend =splits[1], yend = cdf[1]), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[2], y = 0.0, xend = splits[2], yend = max(cdf[1],cdf[2])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[3], y = 0.0, xend = splits[3], yend = max(cdf[2],cdf[3])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[4], y = 0.0, xend = splits[4], yend = max(cdf[3],cdf[4])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[5], y = 0.0, xend = splits[5], yend = max(cdf[4],cdf[5])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[6], y = 0.0, xend = splits[6], yend = max(cdf[5],cdf[6])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[7], y = 0.0, xend = splits[7], yend = max(cdf[6],cdf[7])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[8], y = 0.0, xend = splits[8], yend = max(cdf[7],cdf[8])), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[9], y = 0.0, xend = splits[9], yend = cdf[8]), linetype = "dashed", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[1], y = cdf[1], xend = splits[2], yend = cdf[1]), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[2], y = cdf[2], xend = splits[3], yend = cdf[2]), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[3], y = cdf[3], xend = splits[4], yend = cdf[3]), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[4], y = cdf[4], xend = splits[5], yend = cdf[4]), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[5], y = cdf[5], xend = splits[6], yend = cdf[5]), linetype = "solid", color = "#157595", size = 0.9) +
  geom_segment(aes(x = splits[6], y = cdf[6], xend = splits[7], yend = cdf[6]), linetype = "solid", color = "#157595", size = 0.9) + 
  geom_segment(aes(x = splits[7], y = cdf[7], xend = splits[8], yend = cdf[7]), linetype = "solid", color = "#157595", size = 0.9) + 
  geom_segment(aes(x = splits[8], y = cdf[8], xend = splits[9], yend = cdf[8]), linetype = "solid", color = "#157595", size = 0.9) 
print(p)
ggsave(filename = "binsplit4.png", path = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/figures", device='png', dpi=400)
