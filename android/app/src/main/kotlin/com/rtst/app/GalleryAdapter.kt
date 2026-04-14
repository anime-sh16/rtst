package com.rtst.app

import android.graphics.Bitmap
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

data class GalleryItem(val original: Bitmap, val stylized: Bitmap, val latencyText: String)

class GalleryAdapter(private val items: MutableList<GalleryItem>) :
    RecyclerView.Adapter<GalleryAdapter.GalleryViewHolder>() {

    // ViewHolder: grabs references to the views in item_gallery.xml
    class GalleryViewHolder(view: View) : RecyclerView.ViewHolder(view) {

        val imageOriginal: ImageView = view.findViewById(R.id.imageOriginal)
        val imageStylized: ImageView = view.findViewById(R.id.imageStylized)
        val textLatency: TextView = view.findViewById(R.id.textLatency)

    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): GalleryViewHolder {

        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_gallery, parent, false)
        return GalleryViewHolder(view)

    }

    override fun onBindViewHolder(holder: GalleryViewHolder, position: Int) {

        val item = items[position]
        holder.imageOriginal.setImageBitmap(item.original)
        holder.imageStylized.setImageBitmap(item.stylized)
        holder.textLatency.text = item.latencyText

    }

    override fun getItemCount(): Int = items.size
}
