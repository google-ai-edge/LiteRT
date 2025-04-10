/*
 * Copyright (C) 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.litert.example.classification.ui

import android.content.Context
import android.support.v7.recyclerview.extensions.ListAdapter
import android.support.v7.util.DiffUtil
import android.support.v7.widget.RecyclerView
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import com.google.ai.edge.litert.example.classification.R
import com.google.ai.edge.litert.example.classification.viewmodel.Recognition

class RecognitionAdapter(private val ctx: Context) :
  ListAdapter<Recognition, RecognitionViewHolder>(RecognitionDiffUtil()) {

  /** Inflating the ViewHolder with recognition_item layout and data binding */
  override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecognitionViewHolder {
    val inflater = LayoutInflater.from(ctx)
    val view = inflater.inflate(R.layout.recognition_item, parent, false)
    return RecognitionViewHolder(view)
  }

  // Binding the data fields to the RecognitionViewHolder
  override fun onBindViewHolder(holder: RecognitionViewHolder, position: Int) {
    holder.bindTo(getItem(position))
  }

  private class RecognitionDiffUtil : DiffUtil.ItemCallback<Recognition>() {
    override fun areItemsTheSame(oldItem: Recognition, newItem: Recognition): Boolean {
      return oldItem.label == newItem.label
    }

    override fun areContentsTheSame(oldItem: Recognition, newItem: Recognition): Boolean {
      return oldItem.confidence == newItem.confidence
    }
  }
}

class RecognitionViewHolder(private val view: View) : RecyclerView.ViewHolder(view) {

  private val name = view.findViewById<TextView>(R.id.recognitionName)
  private val prob = view.findViewById<TextView>(R.id.recognitionProb)

  // Binding all the fields to the view - to see which UI element is bind to which field, check
  // out layout/recognition_item.xml
  fun bindTo(recognition: Recognition) {
    name.text = recognition.label
    prob.text = recognition.probabilityString
  }
}
