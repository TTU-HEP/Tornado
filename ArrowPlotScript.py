def create_arrow_plot(finaldf, df3, images_path):
    x = np.array(finaldf['X'])
    y = np.array(finaldf['Y'])
    angles = np.array(finaldf['Angles'])
    r = np.array(finaldf['r'])

    fig, ax = plt.subplots()

    U = [r_value * math.cos(math.radians(-angle)) for angle, r_value in zip(angles, r)]
    V = [r_value * math.sin(math.radians(-angle)) for angle, r_value in zip(angles, r)]

    for i in range(len(U)):
        ax.quiver(x[i], y[i], U[i], V[i], scale=110, color='b', width=0.008)

    for i, label in enumerate(df3['ImageNumber']):
        plt.annotate(label, (x[i], y[i]), color='red', weight='bold', ha='center')

    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 240)
    #x_min, x_max = np.min(x), np.max(x)                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    #y_min, y_max = np.min(y), np.max(y)                                                                                                                                                                                                                                                                                                                                                                                                                                                     

# Include vector lengths in the limits to avoid cropping                                                                                                                                                                                                                                                                                                                                                                                                                                     
    #x_min = min(x_min, np.min(x + U))  # Consider the farthest left point (data + vector)                                                                                                                                                                                                                                                                                                                                                                                                   
    #x_max = max(x_max, np.max(x + U))  # Consider the farthest right point (data + vector)                                                                                                                                                                                                                                                                                                                                                                                                  
    #y_min = min(y_min, np.min(y + V))  # Consider the bottommost point (data + vector)                                                                                                                                                                                                                                                                                                                                                                                                      
    #y_max = max(y_max, np.max(y + V))  # Consider the topmost point (data + vector)                                                                                                                                                                                                                                                                                                                                                                                                         

# Add some padding to the plot limits to ensure nothing is clipped                                                                                                                                                                                                                                                                                                                                                                                                                           
    #x_padding = 0.05 * (x_max - x_min)  # 5% padding on both sides                                                                                                                                                                                                                                                                                                                                                                                                                          
    #y_padding = 0.05 * (y_max - y_min)  # 5% padding on both sides                                                                                                                                                                                                                                                                                                                                                                                                                          

# Set the adjusted limits                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    #ax.set_xlim(x_min - x_padding, x_max + x_padding)                                                                                                                                                                                                                                                                                                                                                                                                                                       
    #ax.set_ylim(y_min - y_padding, y_max + y_padding)                                                                                                                                                                                                                                                                                                                                                                                                                                       
    ax.set_xticks([])
    ax.set_yticks([])

   # reference_arrow = finaldf[finaldf['r'] == finaldf['r'].max()].index[0]                                                                                                                                                                                                                                                                                                                                                                                                                  
   # reference_arrow1 = finaldf[finaldf['r'] == finaldf['r'].min()].index[0]                                                                                                                                                                                                                                                                                                                                                                                                                 


   # arrow_length = math.sqrt(U[reference_arrow]**2 + V[reference_arrow]**2)                                                                                                                                                                                                                                                                                                                                                                                                                 
   # arrow_length1 = math.sqrt(U[reference_arrow1]**2 + V[reference_arrow1]**2)                                                                                                                                                                                                                                                                                                                                                                                                              

    #ax.quiver(x[reference_arrow], y[reference_arrow], U[reference_arrow], V[reference_arrow], scale=110, color='g', width=0.008)                                                                                                                                                                                                                                                                                                                                                            
    #ax.quiver(x[reference_arrow1], y[reference_arrow1], U[reference_arrow1], V[reference_arrow1], scale=110, color='r', width=0.008)                                                                                                                                                                                                                                                                                                                                                        

    #bottom_left_x, bottom_left_y = -40, 180                                                                                                                                                                                                                                                                                                                                                                                                                                                 
   # bottom_left_x1, bottom_left_y1 = -30, 225                                                                                                                                                                                                                                                                                                                                                                                                                                               
   # ax.quiver(bottom_left_x, bottom_left_y, arrow_length, 0, scale=110, color='g', width=0.008)                                                                                                                                                                                                                                                                                                                                                                                             
    #ax.quiver(bottom_left_x1, bottom_left_y1, arrow_length1, 0, scale=110, color='g', width=0.008)                                                                                                                                                                                                                                                                                                                                                                                          


    #ax.annotate(f'Length: {(arrow_length) * 2 / 103:.2f}mm', (bottom_left_x + arrow_length, bottom_left_y + 5), color='red', fontsize=8, weight='bold', ha='center')                                                                                                                                                                                                                                                                                                                        
    #ax.annotate(f'Length: {(arrow_length1) * 2 / 103:.2f}mm', (bottom_left_x1 + arrow_length, bottom_left_y1 + 5), color='green', fontsize=8, weight='bold', ha='center')                                                                                                                                                                                                                                                                                                                   


    reference_arrow = finaldf[finaldf['r'] == finaldf['r'].max()].index[0]
    reference_arrow1 = finaldf[finaldf['r'] == finaldf['r'].min()].index[0]
    average_r = finaldf['r'].mean()
    reference_arrow2 = (finaldf['r'] - average_r).abs().idxmin()



    arrow_length = math.sqrt(U[reference_arrow]**2 + V[reference_arrow]**2)
    arrow_length1 = math.sqrt(U[reference_arrow1]**2 + V[reference_arrow1]**2)
    arrow_length2 = math.sqrt(U[reference_arrow2]**2 + V[reference_arrow2]**2)

    ax.quiver(x[reference_arrow], y[reference_arrow], U[reference_arrow], V[reference_arrow], scale=110, color='g', width=0.008)
    ax.quiver(x[reference_arrow1], y[reference_arrow1], U[reference_arrow1], V[reference_arrow1], scale=110, color='black', width=0.008)
    ax.quiver(x[reference_arrow2], y[reference_arrow2], U[reference_arrow2], V[reference_arrow2], scale=110, color='y', width=0.008)


    bottom_left_x, bottom_left_y = -40, 180
    bottom_left_x1, bottom_left_y1 = -40, 195
    bottom_left_x2, bottom_left_y2 = -40, 220
    ax.quiver(bottom_left_x, bottom_left_y, arrow_length, 0, scale=110, color='g', width=0.008)
    ax.quiver(bottom_left_x1, bottom_left_y1, arrow_length1, 0, scale=110, color='black', width=0.008)
    ax.quiver(bottom_left_x1, bottom_left_y2, arrow_length2, 0, scale=110, color='y', width=0.008)



    ax.annotate(f'Length(Longest): {(arrow_length) * 2 / 103:.2f}mm', (bottom_left_x + arrow_length, bottom_left_y + 5), color='red', fontsize=8, weight='bold', ha='center')
    ax.annotate(f'Length(Smallest): {(arrow_length1) * 2 / 103:.2f}mm', (bottom_left_x1 + arrow_length, bottom_left_y1 + 5), color='red', fontsize=8, weight='bold', ha='center')
    ax.annotate(f'Length(Mean): {(arrow_length2) * 2 / 103:.2f}mm', (bottom_left_x2 + arrow_length, bottom_left_y2 + 5), color='red', fontsize=8, weight='bold', ha='center')


   # reference_arrow = finaldf[finaldf['r'] == finaldf['r'].max()].index[0]                                                                                                                                                                                                                                                                                                                                                                                                                  

   # arrow_length = math.sqrt(U[reference_arrow]**2 + V[reference_arrow]**2)                                                                                                                                                                                                                                                                                                                                                                                                                 

    #ax.quiver(x[reference_arrow], y[reference_arrow], U[reference_arrow], V[reference_arrow], scale=110, color='g', width=0.008)                                                                                                                                                                                                                                                                                                                                                            

    #bottom_left_x, bottom_left_y = -40, 180  # Adjust these values as needed                                                                                                                                                                                                                                                                                                                                                                                                                
    #ax.quiver(bottom_left_x, bottom_left_y, arrow_length, 0, scale=110, color='g', width=0.008)                                                                                                                                                                                                                                                                                                                                                                                             

    #ax.annotate(f'Length: {(arrow_length) * 2 / 103:.2f}mm', (bottom_left_x + arrow_length, bottom_left_y + 5), color='red', fontsize=8, weight='bold', ha='center')                                                                                                                                                                                                                                                                                                                        

    ax.set_title('Combined Arrowplots')
    ax.grid(True)
    plt.tight_layout()
    save_path = os.path.join(images_path, "Arrowplot.jpg")
    plt.savefig(save_path, dpi=100)
    plt.show()
